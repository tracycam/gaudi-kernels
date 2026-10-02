"""Resident identical-arithmetic ABBA/BAAB plus sham; no numerical gate bypass."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from gaudi_kernels.engine.context import context as execution_context
from gaudi_kernels.engine.selection import PolicySelection
from gaudi_kernels.serving.diagnostics.host_clock import clock_domain
from tools.consolidation.analyze_latency_repeat import analyze
from tools.validation.executor.native_service_test import install_step_timer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qualified-result', type=Path, required=True)
    parser.add_argument('--qualified-sha256', required=True)
    parser.add_argument('--cycles', type=int, choices=(2, 3, 4), default=2)
    parser.add_argument('--experiment', choices=('migration', 'runtime'), default='migration')
    parser.add_argument('--device-profile', action='store_true')
    args = parser.parse_args()
    data = args.qualified_result.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.qualified_sha256:
        raise ValueError('Qualified source result bytes changed')
    qualified = json.loads(data)
    if qualified.get('status') != 'PASS' or qualified.get('candidate_accepted') is not True:
        raise ValueError('Requires an already accepted full-model numerical case')
    boundaries = qualified.get('boundary_hashes', {})
    if (boundaries.get('pass') is not True or boundaries.get('records_compared') != 6696
            or qualified.get('fp32_quality', {}).get('passed') is not True):
        raise ValueError('Requires complete producer-boundary evidence')
    document = qualified['migration_abba'][0]['resolved_selection']
    selected = PolicySelection.from_dict(document)
    document = selected.to_dict()
    document['engine']['runtime']['host'] = execution_context().selection.engine.runtime.host.__dict__
    document = PolicySelection.from_dict(document).to_dict()
    if selected.reference != 'device' or execution_context().startup.layers != 70:
        raise ValueError('Full70 device arithmetic required')
    if execution_context().native.boundary_positions:
        raise ValueError('No diagnostic tensor readbacks in timed repeat')
    candidates = [r for r in qualified['runs'] if r['name'].endswith('native-policy-abba-0')]
    if len(candidates) != 1 or len(candidates[0]['ids']) != 160:
        raise ValueError('Expected qualified 160-token sequence')
    expected = candidates[0]['ids']
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    result = {'status': 'RUNNING', 'candidate_accepted': False, 'context': 4096,
              'qualified_result': str(args.qualified_result), 'qualified_sha256': args.qualified_sha256,
              'config': qualified['config'], 'selection': document, 'clock_domain': clock_domain(),
              'runs': [], 'scope': 'Targeted performance diagnostic; reuses prior numerical evidence, no new model admission'}
    def save():
        temporary = args.out/'result.json.tmp'
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(args.out/'result.json')
    save()
    llm = None
    try:
        begin = time.monotonic()
        from gaudi_kernels.serving.host_placement import vllm_kwargs
        model_config = dict(qualified['config']) | vllm_kwargs()
        result['config'] = model_config
        llm = LLM(**model_config)
        result['init_s'] = time.monotonic()-begin
        tokenizer = llm.get_tokenizer()
        prefix = tokenizer.encode('Record: the verification code for the copper lantern is 7391.\n', add_special_tokens=False)
        filler = tokenizer.encode('This document describes reproducible experiments and laboratory observations.\n', add_special_tokens=False)
        suffix = tokenizer.encode('\nQuestion: What is the verification code for the copper lantern? State the code, explain the evidence, and describe how to verify it.\nAnswer:', add_special_tokens=False)
        size = result['context']-len(prefix)-len(suffix)
        prompt = {'prompt_token_ids': prefix+(filler*((size+len(filler)-1)//len(filler)))[:size]+suffix}
        steps, counts = [], {}
        install_step_timer(llm.llm_engine, steps, counts)
        def add_request(prompt, params, lora_request=None, priority=0):
            params.output_kind = RequestOutputKind.CUMULATIVE
            return llm.llm_engine.add_request(str(next(llm.request_counter)), prompt, params,
                                            lora_request=lora_request, priority=priority)
        llm._add_request = add_request
        schedule = []
        for cycle in range(args.cycles):
            order = ('historical', 'canonical', 'canonical', 'historical') if cycle%2 == 0 else ('canonical', 'historical', 'historical', 'canonical')
            schedule.extend(('paired', cycle, arm, installer) for arm, installer in enumerate(order))
        if args.experiment == 'runtime':
            schedule = []
            for cycle in range(args.cycles):
                order = ('legacy', 'compact', 'compact', 'legacy') if cycle%2 == 0 else ('compact', 'legacy', 'legacy', 'compact')
                schedule.extend(('runtime', cycle, arm, mode) for arm, mode in enumerate(order))
            schedule.extend(('diagnostic', args.cycles, i, mode) for i, mode in enumerate(('detail', 'device')))
        else:
            schedule.extend(('sham', args.cycles, arm, 'historical') for arm in range(4))
        for kind, cycle, arm, installer in schedule:
            mode = installer if args.experiment == 'runtime' else 'compact'
            actual_installer = 'canonical' if args.experiment == 'runtime' else installer
            llm.collective_rpc('native_configure', args=(False,))
            method = 'production_policy_reference' if actual_installer == 'historical' else 'production_policy'
            changed = llm.collective_rpc(method, args=(document,))
            if len(changed) != 8 or any(r.get('resolved_config') != document or r['details']['selected_layer_count'] != 70 for r in changed):
                raise ValueError('Installer selection or layer coverage differs')
            steps.clear(); counts.clear()
            warm = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=160, ignore_eos=True), use_tqdm=False)[0].outputs[0]
            warm_record = {'kind': kind, 'cycle': cycle, 'arm': arm, 'installer': installer,
                           'ids': list(warm.token_ids), 'text': warm.text, 'steps': list(steps)}
            result.setdefault('preconditioning', []).append(warm_record)
            save()
            if warm_record['ids'] != expected or '7391' not in warm.text:
                raise ValueError('Bridge preconditioning differs from qualified token sequence')
            llm.llm_engine.engine_core.call_utility('runtime_timing_take')
            llm.collective_rpc('native_configure', args=(True, True, True, mode))
            steps.clear(); counts.clear()
            output = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=160, ignore_eos=True), use_tqdm=False)[0].outputs[0]
            ranks = llm.collective_rpc('native_summary')
            row = {'kind': kind, 'cycle': cycle, 'arm': arm, 'installer': installer,
                   'ids': list(output.token_ids), 'text': output.text, 'steps': list(steps),
                   'ranks': ranks, 'installer_records': changed}
            row['replay_mode'] = mode
            row['core_steps'] = llm.llm_engine.engine_core.call_utility('runtime_timing_take')
            row['host_placement'] = llm.collective_rpc('host_placement_snapshot')
            row['execution_gate_pass'] = (row['ids'] == expected and '7391' in output.text
                and len(ranks) == 8 and all(r['native_steps'] and r['single_rpc_steps']
                    and r['captures'] and all(c['status'] == 'NUMERICAL_REPLAY_PASS' for c in r['captures']) for r in ranks))
            result['runs'].append(row)
            save()
            if not row['execution_gate_pass']:
                raise ValueError('Native sequence or capture coverage failed')
            print('LATENCY_REPEAT_ARM', json.dumps({k: row[k] for k in ('kind', 'cycle', 'arm', 'installer', 'execution_gate_pass')}), flush=True)
        if args.device_profile:
            llm.collective_rpc('native_configure', args=(True, True, True, 'compact'))
            armed = llm.collective_rpc('native_profile', args=(3,))
            if not all(r['armed'] for r in armed):
                raise RuntimeError('Device profiler admission failed')
            profiled = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=16, ignore_eos=True), use_tqdm=False)[0].outputs[0]
            result['device_profile'] = {'ids': list(profiled.token_ids), 'ranks': llm.collective_rpc('native_summary'),
                                        'scope': 'Separate diagnostic; excluded from every throughput comparison'}
            if result['device_profile']['ids'] != expected[:16]:
                raise ValueError('Profiled token sequence differs')
            save()
        if args.experiment == 'runtime':
            from tools.consolidation.analyze_runtime import analyze as analyze_runtime
            report = analyze_runtime(result)
        else:
            report = analyze(result)
        (args.out/'attribution.json').write_text(json.dumps(report, indent=2)+'\n')
        result['performance_diagnostic_pass'] = report.get('all_paired_cycles_pass')
        result['status'] = 'DIAGNOSTIC'
    except BaseException:
        result['status'] = 'FAIL'
        result['error'] = traceback.format_exc()
        raise
    finally:
        if llm is not None:
            try:
                result['cleanup'] = llm.collective_rpc('native_configure', args=(False,))
            except BaseException:
                result['cleanup_error'] = traceback.format_exc()
        save()


if __name__ == '__main__':
    main()
