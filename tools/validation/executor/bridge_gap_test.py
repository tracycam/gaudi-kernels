"""Same resident model/recipes: bridge versus the retained SDK replay control.

Diagnostic only. The legacy recorder is loaded in both arms; it is not a
production backend proposal or an independent model quality qualification.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

from gaudi_kernels.engine.context import context
from tools.validation.executor.step_timing import install_step_timer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--tokens', type=int, default=160)
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--isolate-rpc', action='store_true',
                        help='Add native replay with the ordinary execute/sample RPC split')
    args = parser.parse_args()
    runtime = context()
    if runtime.selection.engine.runtime.executor != 'legacy_native':
        raise ValueError('Diagnostic requires the retained control in both arms')
    if not 24 <= args.tokens <= 160 or not 128 <= args.context <= 4096:
        raise ValueError('Bounded comparison only')
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from gaudi_kernels.serving.host_placement import vllm_kwargs
    options = dict(model='/data/models/MiMo-V2.6-Pro-RL', trust_remote_code=True,
                   tensor_parallel_size=8, enable_expert_parallel=False, dtype='bfloat16',
                   enforce_eager=False, async_scheduling=False, block_size=128,
                   max_model_len=8192, max_num_seqs=2, gpu_memory_utilization=.95,
                   disable_log_stats=True, enable_prefix_caching=False, seed=0,
                   limit_mm_per_prompt={'audio': 0, 'image': 0, 'video': 0})
    options.update(vllm_kwargs(diagnostics=True))
    if runtime.startup.layers != 70:
        cfg = json.loads((Path(options['model'])/'config.json').read_text())
        n = runtime.startup.layers
        options.update(hf_overrides=dict(num_hidden_layers=n,
            hybrid_layer_pattern=cfg['hybrid_layer_pattern'][:n], moe_layer_freq=cfg['moe_layer_freq'][:n]),
            gpu_memory_utilization=.3, num_gpu_blocks_override=256)
    result = dict(status='RUNNING', layers=runtime.startup.layers, options=options, runs=[],
                  scope='Same-process causal diagnosis; no independent language-quality qualification')

    def save():
        temporary = args.out/'result.json.tmp'
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(args.out/'result.json')

    save()
    start = time.monotonic()
    llm = LLM(**options)
    result['load_seconds'] = time.monotonic()-start
    steps, counts = [], {}
    install_step_timer(llm.llm_engine, steps, counts)

    def add_request(prompt, params, lora_request=None, priority=0):
        params.output_kind = RequestOutputKind.CUMULATIVE
        return llm.llm_engine.add_request(str(next(llm.request_counter)), prompt, params,
                                         lora_request=lora_request, priority=priority)

    llm._add_request = add_request
    tokenizer = llm.get_tokenizer()
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    repeated = encode(' Describe software architecture.')
    prefix = encode('Record: the verification code for the copper lantern is 7391.\n')
    filler = encode('This document describes reproducible experiments and laboratory observations.\n')
    suffix = encode('\nQuestion: What is the verification code for the copper lantern? State the code, explain the evidence, and describe how to verify it.\nAnswer:')
    n = args.context-len(prefix)-len(suffix)
    if n <= 0:
        raise ValueError('Context too short for retrieval fixture')
    prompts = dict(repeated=(repeated*((args.context+len(repeated)-1)//len(repeated)))[:args.context],
                   retrieval=prefix+(filler*((n+len(filler)-1)//len(filler)))[:n]+suffix)
    result['prompt_ids'] = prompts
    sample = SamplingParams(temperature=0, max_tokens=args.tokens, ignore_eos=True)
    try:
        for fixture, prompt_ids in prompts.items():
            prompt = {'prompt_token_ids': prompt_ids}
            baseline = None
            schedule = ((False, False), (True, True), (True, True), (False, False))
            if args.isolate_rpc:
                schedule = ((False, False), (True, False), (True, True),
                            (True, True), (True, False), (False, False))
            for arm, (active, single_rpc) in enumerate(schedule):
                # Warm the complete position range identically before every arm.
                llm.collective_rpc('native_configure', args=(False,))
                warm = llm.generate([prompt], sample, use_tqdm=False)[0].outputs[0]
                warm_ids = list(warm.token_ids)
                if baseline is None:
                    baseline = warm_ids
                if warm_ids != baseline:
                    raise AssertionError('Bridge warm sequence changed')
                llm.collective_rpc('native_configure', args=(active, single_rpc, active))
                steps.clear()
                counts.clear()
                output = llm.generate([prompt], sample, use_tqdm=False)[0].outputs[0]
                ids = list(output.token_ids)
                row = dict(fixture=fixture, arm=arm, active=active, single_rpc=single_rpc, ids=ids, text=output.text,
                           token_match=ids==baseline, steps=list(steps))
                row['native'] = llm.collective_rpc('native_summary')
                row['framework'] = llm.collective_rpc('framework_execution_snapshot')
                row['placement'] = llm.collective_rpc('host_placement_snapshot')
                native_used = len(row['native']) == 8 and all(rank['native_steps'] for rank in row['native'])
                row['native_used_all_ranks'] = bool(native_used)
                row['capture_gate_pass'] = all(c['status']=='NUMERICAL_REPLAY_PASS'
                    for rank in row['native'] for c in rank['captures'])
                lo = min(32, args.tokens//2)
                steady = [s for s in steps if s['new_tokens']==1 and lo<=s['emitted']<args.tokens]
                if not steady:
                    raise AssertionError('No comparable resident window')
                row.update(tps=len(steady)*1e9/(steady[-1]['end_ns']-steady[0]['begin_ns']),
                           p50_ms=statistics.median(s['ms'] for s in steady), window_steps=len(steady))
                result['runs'].append(row)
                save()
                print('GAP_ARM', json.dumps({k:row[k] for k in ('fixture','arm','active','tps','p50_ms','token_match','native_used_all_ranks','capture_gate_pass')}), flush=True)
                if not row['token_match'] or not row['capture_gate_pass'] or (active and not native_used):
                    raise AssertionError('Same-computation execution comparison failed')
            if args.diagnose:
                for active in (False, True):
                    llm.collective_rpc('native_configure', args=(active, active, active))
                    llm.collective_rpc('framework_timing', args=(True, False, False))
                    try:
                        llm.generate([prompt], SamplingParams(temperature=0, max_tokens=64, ignore_eos=True), use_tqdm=False)
                    finally:
                        observation = llm.collective_rpc('framework_timing', args=(False,))
                    result.setdefault('diagnostics', []).append(dict(fixture=fixture, active=active,
                        host=observation, native=llm.collective_rpc('native_summary')))
                    save()
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        raise
    finally:
        save()
        llm.collective_rpc('native_configure', args=(False,))


if __name__ == '__main__':
    main()
