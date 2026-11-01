"""Same ordinary model program for upstream/fork wheels; no executor-control RPCs."""
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
    parser.add_argument('--model', type=Path, default=Path('/data/models/MiMo-V2.6-Pro-RL'),
                        help='Checkpoint directory; explicit for machine migration')
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--prompt-fixture', choices=('architecture', 'retrieval'), default='architecture',
                        help='Retrieval additionally checks the answer, independently of baseline equality')
    parser.add_argument('--batch-sizes', default='1,2')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--production-worker', action='store_true',
                        help='Use the ordinary worker; numerical probes install only after timing')
    parser.add_argument('--step-abba', action='store_true',
                        help='Compare forward/whole-step/whole-step/forward in one resident model')
    parser.add_argument('--event-batch-abba', action='store_true',
                        help='Complete-step graph in all arms; only batch internal event publications')
    parser.add_argument('--input-abba', action='store_true',
                        help='Compare SHA-pinned old input traversal to the installed bridge implementation')
    parser.add_argument('--compiled-abba', action='store_true',
                        help='Compare prepared regional submissions to ordinary recipe launchers')
    parser.add_argument('--input-layout-abba', action='store_true',
                        help='Prepared bridge in all arms; compare original/prepared host input layouts')
    parser.add_argument('--host-policy-abba', choices=('affinity', 'utilization', 'critical', 'physical', 'both'),
                        help='Reversible CPU policy ABBA; no device or global governor changes')
    parser.add_argument('--host-policy-control', choices=('utilization',),
                        help='Use the all-thread QoS policy as ABBA control, instead of the original policy')
    parser.add_argument('--diagnostic-host-policy', choices=('utilization', 'critical'),
                        help='Explicit CPU policy for separate post-timing traces; restored before quality checks')
    parser.add_argument('--require-prepared-coverage', action='store_true',
                        help='Require all admitted serial collectives to use prepared adapters')
    parser.add_argument('--report-prepared-coverage', action='store_true',
                        help='Record actual adapter coverage without claiming it passes the coverage gate')
    parser.add_argument('--step-quality', action='store_true',
                        help='Separately compare full logits before/at a 128-token boundary')
    parser.add_argument('--sdk-profile-library', type=Path,
                        help='Direct SDK trace helper, separate diagnostic after timing')
    parser.add_argument('--prequeue-library', type=Path,
                        help='Bounded device gate, separate from TPS measurements')
    parser.add_argument('--submission-timing', action='store_true',
                        help='Separate bounded CPU API timing after performance requests')
    parser.add_argument('--diagnose', action='store_true',
                        help='After all timing runs, collect bounded nested host-call timings')
    parser.add_argument('--cpu-trace', action='store_true',
                        help='With --diagnose, export standard Torch CPU and bridge function traces')
    parser.add_argument('--device-trace', action='store_true',
                        help='With --diagnose, capture HPU activity for 15 decode steps after timing')
    args = parser.parse_args()
    from functools import partial
    from tools.validation.executor.worker_diagnostics import call
    rpc = partial(call, production=args.production_worker)
    if args.submission_timing and args.prequeue_library:
        parser.error('Run API timing and prequeue as separate diagnostics')
    if args.host_policy_control and (not args.host_policy_abba or args.host_policy_abba == 'both'):
        parser.error('A CPU control requires one explicit CPU ABBA candidate')
    if sum((args.step_abba, args.event_batch_abba, args.input_abba, args.compiled_abba, args.input_layout_abba, bool(args.host_policy_abba))) > 1:
        parser.error('Choose one independent ABBA variable')
    abba = args.step_abba or args.event_batch_abba or args.input_abba or args.compiled_abba or args.input_layout_abba
    def configure_arm(enabled):
        rpc(llm, 'framework_step_graph', args=(True if args.event_batch_abba or args.input_abba or args.compiled_abba or args.input_layout_abba else enabled,))
        if args.input_layout_abba:
            rpc(llm, 'framework_compiled_regions', args=(True,))
            rpc(llm, 'framework_input_layout', args=(enabled,))
        if args.compiled_abba:
            rpc(llm, 'framework_compiled_regions', args=(enabled,))
        if args.event_batch_abba:
            rpc(llm, 'framework_event_batching', args=(enabled,))
        if args.input_abba:
            rpc(llm, 'framework_event_batching', args=(False,))
            rpc(llm, 'framework_input_traversal', args=(enabled,))
    if (args.cpu_trace or args.device_trace) and not args.diagnose:
        parser.error('--cpu-trace/--device-trace require --diagnose')
    runtime = context()
    if args.require_prepared_coverage and not (args.compiled_abba or args.input_layout_abba):
        if abba or runtime.selection.engine.runtime.decode_graph != 'step':
            parser.error('Prepared coverage requires a prepared-region A/B or configured step graph')
    if args.prompt_fixture == 'retrieval' and runtime.startup.layers != 70:
        parser.error('Retrieval quality requires the complete 70-layer model')
    if runtime.selection.engine.runtime.executor != 'pytorch':
        raise ValueError('This baseline must not use an external execution engine')
    batches = tuple(int(value) for value in args.batch_sizes.split(','))
    if not batches or any(value < 1 or value > 8 for value in batches):
        parser.error('batch sizes must be 1..8')
    if not 16 <= args.tokens <= 512 or not 1 <= args.rounds <= 4:
        parser.error('bounded tokens/rounds required')
    from vllm import LLM, SamplingParams
    from gaudi_kernels.serving.host_placement import vllm_kwargs
    options = dict(model=str(args.model), trust_remote_code=True,
                   tensor_parallel_size=8, enable_expert_parallel=False, dtype='bfloat16',
                   enforce_eager=False, async_scheduling=False, block_size=128,
                   max_model_len=max(8192, args.context+args.tokens+128),
                   max_num_seqs=max(batches), gpu_memory_utilization=.95,
                   disable_log_stats=True, enable_prefix_caching=False, seed=0,
                   limit_mm_per_prompt={'audio': 0, 'image': 0, 'video': 0})
    options.update(vllm_kwargs(diagnostics=not args.production_worker))
    if runtime.startup.layers != 70:
        config = json.loads((Path(options['model'])/'config.json').read_text())
        layers = runtime.startup.layers
        options['hf_overrides'] = dict(num_hidden_layers=layers,
            hybrid_layer_pattern=config['hybrid_layer_pattern'][:layers],
            moe_layer_freq=config['moe_layer_freq'][:layers])
        options['gpu_memory_utilization'] = .3
        options['num_gpu_blocks_override'] = 256
    result = dict(status='RUNNING', model_layers=runtime.startup.layers, options=options, runs=[],
                  scope='Ordinary PyTorch model path; partial layers never qualify model TPS')

    def save():
        temporary = args.out/'result.json.tmp'
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(args.out/'result.json')

    save()
    begin = time.monotonic()
    llm = LLM(**options)
    result['load_seconds'] = time.monotonic()-begin
    steps, counts = [], {}
    install_step_timer(llm.llm_engine, steps, counts)
    # Reuse the historical measurement convention: expose each emitted token
    # to the client timer rather than final-only output from LLM.generate.
    from vllm.sampling_params import RequestOutputKind
    def add_request(prompt, params, lora_request=None, priority=0):
        params.output_kind = RequestOutputKind.CUMULATIVE
        return llm.llm_engine.add_request(str(next(llm.request_counter)), prompt, params,
                                         lora_request=lora_request, priority=priority)
    llm._add_request = add_request
    encode = lambda text: llm.get_tokenizer().encode(text, add_special_tokens=False)
    token = encode(' Describe software architecture.')
    prompt_ids = (token*((args.context+len(token)-1)//len(token)))[:args.context]
    if args.prompt_fixture == 'retrieval':
        prefix = encode('Record: the verification code for the copper lantern is 7391.\n')
        filler = encode('This document describes reproducible experiments and laboratory observations.\n')
        suffix = encode('\nQuestion: What is the verification code for the copper lantern? State the code, explain the evidence, and describe how to verify it.\nAnswer:')
        remaining = args.context-len(prefix)-len(suffix)
        if remaining <= 0:
            raise ValueError('Context too short for retrieval fixture')
        prompt_ids = prefix+(filler*((remaining+len(filler)-1)//len(filler)))[:remaining]+suffix
    prompts = [{'prompt_token_ids': list(prompt_ids)} for _ in range(max(batches))]
    result.update(prompt_fixture=args.prompt_fixture, prompt_ids=prompt_ids)
    save()
    sample = SamplingParams(temperature=0, max_tokens=args.tokens, ignore_eos=True)
    # ABBA already warms a full request before each arm. Configured runs also
    # visit the measured page/bucket range before timing, without policy RPCs.
    result['initial_warmup_tokens'] = 16 if abba else args.tokens
    discovered = False
    for batch in batches:
        # Compile/warm separately from measured real requests.
        llm.generate(prompts[:batch], SamplingParams(temperature=0,
                     max_tokens=result['initial_warmup_tokens'], ignore_eos=True),
                     use_tqdm=False)
        if not discovered and (args.host_policy_abba == 'critical' or args.diagnostic_host_policy == 'critical'):
            rpc(llm, 'framework_timing', args=(True, True, False))
            try:
                llm.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=16,
                             ignore_eos=True), use_tqdm=False)
            finally:
                observations = rpc(llm, 'framework_timing', args=(False,))
            traces = {}
            for rank in observations:
                path = Path(rank['cpu_trace'])
                destination = path.with_name(f'{path.stem}-submit-discovery{path.suffix}')
                path.rename(destination)
                traces[str(rank['rank'])] = str(destination)
            result['submission_thread_discovery'] = rpc(llm, 'host_execution_policy', args=(None, traces))
            result['submission_thread_traces'] = traces
            discovered = True
            save()
        baseline_rows = None
        arms = ((False, True, True, False)*(2 if args.host_policy_abba == 'both' else 1)
                if args.host_policy_abba else (False, True, True, False) if abba else (None,)*args.rounds)
        for repeat, step_graph in enumerate(arms):
            host_mode = ('utilization' if repeat < 4 else 'affinity') if args.host_policy_abba == 'both' else args.host_policy_abba
            host_enabled = step_graph if host_mode else None
            host_policy = rpc(llm, 'host_execution_policy', args=(host_mode if host_enabled else args.host_policy_control,)) if host_mode else None
            input_layout = step_graph if args.input_layout_abba else None
            compiled_regions = True if args.input_layout_abba else step_graph if args.compiled_abba else None
            input_optimized = step_graph if args.input_abba else None
            event_batch = step_graph if args.event_batch_abba else None
            if args.event_batch_abba or args.input_abba or args.compiled_abba or args.input_layout_abba or args.host_policy_abba:
                step_graph = True
            if step_graph is not None:
                rpc(llm, 'framework_step_graph', args=(step_graph,))
                if args.input_abba:
                    configure_arm(input_optimized)
                if args.input_layout_abba:
                    configure_arm(input_layout)
                llm.generate(prompts[:batch], sample, use_tqdm=False)
                if event_batch is not None:
                    rpc(llm, 'framework_event_batching', args=(event_batch,))
                if compiled_regions is not None:
                    rpc(llm, 'framework_compiled_regions', args=(compiled_regions,))
            counts.clear()
            steps.clear()
            begin = time.perf_counter_ns()
            outputs = llm.generate(prompts[:batch], sample, use_tqdm=False)
            end = time.perf_counter_ns()
            rows = [dict(ids=list(output.outputs[0].token_ids), text=output.outputs[0].text)
                    for output in outputs]
            if args.prompt_fixture == 'retrieval' and any('7391' not in row['text'] for row in rows):
                result.update(status='FAIL', failed_outputs=rows, quality_reason='Retrieval answer missing')
                save()
                raise AssertionError('Retrieval generation omitted the verification code')
            if baseline_rows is None:
                baseline_rows = rows
            if (abba or args.host_policy_abba) and rows != baseline_rows:
                result.update(status='FAIL', failed_outputs=rows, expected_outputs=baseline_rows)
                save()
                raise AssertionError('Whole-step output differs from forward-only baseline')
            # Exactly one emitted token per request selects fully resident decode
            # steps. Total tokens/time is retained alongside this subset.
            resident = [step for step in steps if step['known_requests'] == batch and
                        len(step['per_request_new']) == batch and
                        all(value == 1 for value in step['per_request_new']) and
                        step['emitted_min'] >= 16]
            if not resident:
                result.update(status='FAIL', failed_steps=list(steps), failed_outputs=rows)
                save()
                raise AssertionError('No fully resident decode steps')
            result['runs'].append(dict(batch=batch, repeat=repeat, step_graph=step_graph,
                host_policy_mode=host_mode, host_policy_enabled=host_enabled, host_policy=host_policy,
                host_policy_control=args.host_policy_control,
                event_batching=event_batch,
                input_optimized=input_optimized,
                input_layout_optimized=input_layout,
                compiled_regions=compiled_regions,
                ranks=rpc(llm, 'framework_execution_snapshot') if abba or args.require_prepared_coverage or args.report_prepared_coverage else [],
                outputs=rows, steps=list(steps),
                total_tokens=sum(len(row['ids']) for row in rows), wall_ms=(end-begin)/1e6,
                total_tokens_per_second=sum(len(row['ids']) for row in rows)*1e9/(end-begin),
                resident_step_count=len(resident),
                resident_continuous_ms=(resident[-1]['end_ns']-resident[0]['begin_ns'])/1e6,
                resident_continuous_tps=sum(step['new_tokens'] for step in resident)*1e9/
                                        (resident[-1]['end_ns']-resident[0]['begin_ns']),
                resident_tokens_per_second=sum(step['new_tokens'] for step in resident)*1000/
                                           sum(step['ms'] for step in resident),
                resident_p50_ms=statistics.median(step['ms'] for step in resident)))
            save()
            if input_layout and any(not r.get('input_layout_hits') for r in result['runs'][-1]['ranks']):
                result.update(status='FAIL', coverage_failure='Prepared host layout not exercised on every rank')
                save()
                raise AssertionError(result['coverage_failure'])
            if args.require_prepared_coverage and (compiled_regions or not abba):
                ranks = result['runs'][-1]['ranks']
                if len(ranks) != 8 or any(not any(p['stats'].get('prepared_replayed_commands', 0)
                                                 for p in rank['step_plans']) for rank in ranks):
                    result.update(status='FAIL', coverage_failure='Missing complete-step coverage for one or more ranks')
                    save()
                    raise AssertionError(result['coverage_failure'])
                for rank in ranks:
                    if not abba and not rank.get('input_layout_hits'):
                        result.update(status='FAIL', coverage_failure='Configured input layout was not exercised')
                        save()
                        raise AssertionError(result['coverage_failure'])
                    for plan in rank['step_plans']:
                        stats = plan['stats']
                        if (not stats.get('prepared_commands') or stats.get('prepared_rejections') or
                                stats.get('prepared_collectives') != stats['coalesced_collectives']):
                            result.update(status='FAIL', coverage_failure=dict(rank=rank['rank'], stats=stats))
                            save()
                            raise AssertionError('Prepared adapter misses eligible model commands')
        if args.host_policy_abba:
            rpc(llm, 'host_execution_policy', args=(None,))
    if args.diagnostic_host_policy:
        result['diagnostic_host_policy'] = rpc(llm, 'host_execution_policy', args=(args.diagnostic_host_policy,))
        save()
    if args.diagnose:
        # Run only after measured requests. No new executor, device sync or
        # tensor read is inserted by these observers. Restore before snapshot.
        result['host_timings'] = []
        for scope in (False, True) if abba else (None,):
            if scope is not None:
                configure_arm(scope)
            rpc(llm, 'framework_timing', args=(True, args.cpu_trace, args.device_trace))
            try:
                llm.generate(prompts[:1], SamplingParams(temperature=0,
                             max_tokens=16 if args.device_trace else 64, ignore_eos=True),
                             use_tqdm=False)
            finally:
                observations = rpc(llm, 'framework_timing', args=(False,))
                for rank in observations:
                    for kind in ('cpu_trace', 'device_trace'):
                        if rank.get(kind):
                            path = Path(rank[kind])
                            destination = path.with_name(f'{path.stem}-scope-{scope}{path.suffix}')
                            path.rename(destination)
                            rank[kind] = str(destination)
                result['host_timings'].append(dict(step_graph=True if args.event_batch_abba or args.input_abba or args.compiled_abba or args.input_layout_abba else scope,
                    event_batching=scope if args.event_batch_abba else None,
                    input_optimized=scope if args.input_abba else None,
                    input_layout_optimized=scope if args.input_layout_abba else None,
                    compiled_regions=True if args.input_layout_abba else scope if args.compiled_abba else None, ranks=observations))
                save()
    if args.submission_timing:
        result['submission_timings'] = []
        rpc(llm, 'framework_step_graph', args=(True,))
        for enabled in (False, True):
            rpc(llm, 'framework_event_batching', args=(enabled,))
            rpc(llm, 'framework_submission_timing', args=(True,))
            try:
                llm.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=64, ignore_eos=True), use_tqdm=False)
            finally:
                result['submission_timings'].append(dict(event_batching=enabled,
                    ranks=rpc(llm, 'framework_submission_timing', args=(False,))))
                save()
    if args.prequeue_library:
        rpc(llm, 'framework_step_graph', args=(True,))
        rpc(llm, 'framework_prequeue_probe', args=(str(args.prequeue_library),))
        try:
            llm.generate(prompts[:1], SamplingParams(temperature=0,max_tokens=32,ignore_eos=True),use_tqdm=False)
        finally:
            result['prequeue_probe'] = rpc(llm, 'framework_prequeue_probe')
            save()
        if any(len(rank['rows']) != 8 or not all(row['complete_step'] for row in rank['rows'])
               for rank in result['prequeue_probe']):
            raise AssertionError('Incomplete prequeue probe')
    if args.sdk_profile_library:
        result['sdk_profiles'] = []
        for scope in (False, True):
            rpc(llm, 'framework_step_graph', args=(scope,))
            tag = f'sdk-scope-{int(scope)}'
            rpc(llm, 'framework_device_profile', args=(str(args.sdk_profile_library), tag))
            try:
                llm.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=32,
                                                        ignore_eos=True), use_tqdm=False)
            finally:
                ranks = rpc(llm, 'framework_device_profile',
                        args=(str(args.sdk_profile_library), tag, False))
                result['sdk_profiles'].append(dict(step_graph=scope, ranks=ranks))
                save()
            if any(rank['status'] != 'PASS' for rank in ranks):
                raise AssertionError('SDK profile incomplete')
    if args.diagnostic_host_policy:
        result['diagnostic_host_policy_restored'] = rpc(llm, 'host_execution_policy', args=(None,))
        save()
    if args.step_quality:
        import torch
        result['step_quality'] = []
        for batch in batches:
            for length in (2, 129, 130):
                snapshots = []
                for scope in (False, True):
                    configure_arm(scope)
                    tag = f'step-b{batch}-n{length}-s{int(scope)}'
                    captures = rpc(llm, 'production_quality_capture', args=(tag, 'plain'))
                    try:
                        llm.generate(prompts[:batch], SamplingParams(temperature=0,
                                     max_tokens=length, ignore_eos=True), use_tqdm=False)
                    finally:
                        rpc(llm, 'production_quality_capture', args=(None,))
                    path = next(item['path'] for item in captures if item['rank'] == 0)
                    snapshots.append(torch.load(path, map_location='cpu', weights_only=True))
                same = {key: torch.equal(snapshots[0][key], snapshots[1][key])
                        for key in ('input_ids', 'positions', 'logits')}
                result['step_quality'].append(dict(batch=batch, length=length, equal=same,
                    variable='input_layout_optimized' if args.input_layout_abba else 'compiled_regions' if args.compiled_abba else 'input_optimized' if args.input_abba else 'event_batching' if args.event_batch_abba else 'step_graph'))
                save()
                if not all(same.values()):
                    result['status'] = 'FAIL'
                    save()
                    raise AssertionError('Whole-step full-logits quality check failed')
    result['ranks'] = rpc(llm, 'framework_execution_snapshot')
    result['host_placement'] = rpc(llm, 'host_placement_snapshot')
    if any(rank['external_executors'] for rank in result['ranks']):
        raise AssertionError('External execution engine loaded')
    if abba:
        for run in result['runs']:
            if run['step_graph'] and any(not rank['step_plans'] or not rank['step_graph_calls'] or
                    not all(p['stats']['ready'] and not p['stats']['failed'] and p['stats']['replays'] > 0
                            for p in rank['step_plans']) for rank in run['ranks']):
                raise AssertionError('Whole-step graph was not exercised on every rank')
            if run.get('event_batching') and any(not any(
                    p['stats'].get('event_batches',0)>0 for p in rank['step_plans']) for rank in run['ranks']):
                raise AssertionError('Event batching did not execute on every rank')
            if run.get('compiled_regions') and any(not any(
                    p['stats'].get('prepared_region_replays',0)>0 for p in rank['step_plans']) for rank in run['ranks']):
                raise AssertionError('Compiled regions did not execute on every rank')
    result['status'] = 'PASS'
    save()
    print(json.dumps(dict(status=result['status'], layers=result['model_layers'],
                          runs=[{key:run[key] for key in ('batch','repeat','resident_tokens_per_second',
                                                         'total_tokens_per_second')}
                                for run in result['runs']]), indent=2), flush=True)


if __name__ == '__main__':
    main()
