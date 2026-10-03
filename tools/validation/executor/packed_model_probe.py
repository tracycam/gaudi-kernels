"""Real TP model all-query packed-versus-sequential semantic validation."""
import argparse
import json
from pathlib import Path



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layers', type=int, choices=(2, 70), default=2)
    p.add_argument('--model', required=True)
    p.add_argument('--small-only', action='store_true')
    p.add_argument('--moe-row-probe', nargs='+', type=int, choices=(8,12,16,24),
                   help='Component-only complete MoE chain on loaded weights; serving row guards unchanged')
    p.add_argument('--serving-teacher', action='store_true',
                   help='Also compare packed multi-position logits against ordinary vLLM serving')
    p.add_argument('--scheduled-serving', action='store_true',
                   help='Also run the experimental compact consumer of actual scheduling and allocator KV')
    p.add_argument('--static-replay', action='store_true',
                   help='Probe same-address recorded target replay; not explicit graph construction or throughput')
    p.add_argument('--advancing-replay', action='store_true',
                   help='Update recorded token/position/KV bindings across accepted/rejected input prefixes')
    p.add_argument('--target-features', action='store_true',
                   help='Audit real residual-completed target features at independently hooked decoder boundaries')
    p.add_argument('--trace-layers', type=int, nargs='+', default=(),
                   help='Freeze operator input/output boundaries for serving-teacher diagnosis')
    p.add_argument('--native-packed-swa', action='store_true',
                   help='Use request-owned aligned pages and the pinned multi-query SWA kernel')
    p.add_argument('--defer-precision-gate', action='store_true',
                   help='Report floating regression thresholds without blocking finite/KV/causality functional diagnostics')
    p.add_argument('--swa-witness', action='store_true',
                   help='Freeze native SWA operands/output for a small diagnostic, never for timing')
    p.add_argument('--dflash-cycle', action='store_true',
                   help='Run actual checkpoint drafter/target/acceptance/context cycles after target diagnostics')
    p.add_argument('--cycle-prompts', type=Path,
                   help='Private JSON list of prompt_token_ids objects; never published with source')
    p.add_argument('--cycle-steps', type=int, default=4)
    p.add_argument('--cycle-verify-rows', type=int, default=4)
    p.add_argument('--record-dflash-cycle', action='store_true',
                   help='Record the whole fixed-extent cycle after one eager warmup; not serving admission')
    p.add_argument('--cycle-batch-sweep', nargs='+', type=int, choices=(1, 2, 3),
                   help='Reuse one resident target to diagnose these independent request counts')
    p.add_argument('--cycle-row-sweep', nargs='+', type=int, choices=tuple(range(2, 9)),
                   help='Reuse one resident target to diagnose these target verify extents')
    p.add_argument('--cycle-routes', action='store_true',
                   help='Intrusively snapshot actual MoE route IDs; this run cannot establish service TPS')
    p.add_argument('--cycle-binding-sweep', nargs='+', choices=('legacy', 'static'),
                   help='Pair changing-history preparation with fixed-capacity page bindings in one process')
    p.add_argument('--profile-dflash-cycle', action='store_true',
                   help='Attempt direct Synapse rank0 trace on recorded cycle4; profiling is diagnostic only')
    p.add_argument('--cycle-gqa-sweep', nargs='+', choices=('broadcast', 'folded'),
                   help='Compare shared-KV batch broadcasting with query heads folded into M')
    p.add_argument('--cycle-moe-sweep', nargs='+', choices=('production', 'compact12'),
                   help='Private B3T4 cycle ABBA; compact12 never changes serving admission')
    args = p.parse_args()
    if args.dflash_cycle and (args.layers != 70 or args.cycle_prompts is None):
        p.error('Real DFlash cycle requires70 target layers and explicit prompt fixtures')
    if args.record_dflash_cycle and (not args.dflash_cycle or args.cycle_steps < 3):
        p.error('Cycle recording requires --dflash-cycle and at least three cycles')
    if args.profile_dflash_cycle and (not args.record_dflash_cycle or args.cycle_steps < 5):
        p.error('Cycle profiling requires recording and at least five cycles')
    if (args.cycle_batch_sweep or args.cycle_row_sweep or args.cycle_binding_sweep or args.cycle_gqa_sweep or args.cycle_moe_sweep) and not args.dflash_cycle:
        p.error('Cycle sweeps require --dflash-cycle')
    if args.cycle_moe_sweep and (args.cycle_batch_sweep != [3] or args.cycle_row_sweep != [4]):
        p.error('Private compact12 paired cycles require --cycle-batch-sweep3 --cycle-row-sweep4')
    if args.dflash_cycle:
        prompts = json.loads(args.cycle_prompts.read_text())
        if args.cycle_batch_sweep and max(args.cycle_batch_sweep) > len(prompts):
            p.error('Cycle sweep needs a real prompt fixture per independent request')
    from gaudi_kernels.engine.context import context
    from gaudi_kernels.serving.host_placement import vllm_kwargs
    from vllm import LLM, SamplingParams
    if context().startup.layers != args.layers:
        raise ValueError('Explicit layer scope mismatch')
    opts = dict(model=args.model, trust_remote_code=True, tensor_parallel_size=8,
                enable_expert_parallel=False, dtype='bfloat16', enforce_eager=False,
                async_scheduling=False, max_model_len=8192, block_size=128, max_num_seqs=3,
                gpu_memory_utilization=.3 if args.layers == 2 else .95,
                max_num_batched_tokens=512, enable_prefix_caching=False, seed=0,
                limit_mm_per_prompt={'audio': 0, 'image': 0, 'video': 0})
    opts.update(vllm_kwargs())
    if args.layers == 2:
        cfg = json.loads((Path(args.model)/'config.json').read_text())
        opts.update(hf_overrides={'num_hidden_layers': 2, 'hybrid_layer_pattern': cfg['hybrid_layer_pattern'][:2],
                                 'moe_layer_freq': cfg['moe_layer_freq'][:2]}, num_gpu_blocks_override=256)
    llm = LLM(**opts)
    llm.collective_rpc('native_configure', args=(False, False, False, 'compact'))
    if args.moe_row_probe:
        ranks = llm.collective_rpc('moe_rows_probe', args=({'rows': args.moe_row_probe},))
        result = dict(pass_=len(ranks) == 8 and all(r['pass'] for r in ranks), ranks=ranks)
        result['pass'] = result.pop('pass_')
        (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        if not result['pass']:
            raise RuntimeError('Complete MoE row coverage failed')
        return
    plan = {'cases': [{'name': 'ragged-small', 'rows': [1, 4, 8], 'starts': [127, 126, 0], 'commits': [1, 2, 8]}]}
    plan['native_swa'] = args.native_packed_swa
    plan['defer_precision_gate'] = args.defer_precision_gate
    plan['swa_witness'] = args.swa_witness
    if args.trace_layers:
        if any(i < 0 or i >= args.layers for i in args.trace_layers) or not args.serving_teacher:
            raise ValueError('Boundary tracing requires valid decoder indices and serving teacher')
        plan['trace_layers'] = tuple(args.trace_layers)
    if not args.small_only:
        plan['cases'].append({'name': 'ragged-133', 'rows': [1, 4, 128], 'starts': [127, 126, 0], 'commits': [0, 2, 128]})
    if args.serving_teacher:
        from tools.validation.executor.batch_quality import generate_with_ids, attach_owners
        base = llm.get_tokenizer().encode('The quick brown fox jumps over the lazy dog. ')
        prompts = [{'prompt_token_ids': (base * 128)[:125 + i]} for i in range(3)]
        llm.collective_rpc('production_quality_capture', args=('packed-serving-control', 'batch', 3))
        outputs, mapping = generate_with_ids(llm, prompts,
            SamplingParams(temperature=0, max_tokens=5, ignore_eos=True))
        ranks = llm.collective_rpc('production_quality_capture', args=(None,))
        frames = next(rank['captured_forwards'] for rank in ranks if rank['rank'] == 0)
        attach_owners(frames, outputs, mapping)
        rows = [[] for _ in prompts]
        for frame in frames:
            for logit_row, (owner, query) in enumerate(zip(frame['request_indices'], frame['logits_indices'])):
                if owner is not None:
                    rows[owner].append({'position': frame['positions'][query], 'token': frame['input_ids'][query],
                                        'path': frame['path'], 'row': logit_row})
        for index, values in enumerate(rows):
            values.sort(key=lambda value: value['position'])
            start = len(prompts[index]['prompt_token_ids'])
            if [value['position'] for value in values] != list(range(start, start + 4)):
                raise ValueError('Incomplete ordinary serving teacher query coverage')
        plan['cases'].append({'name': 'serving-teacher', 'rows': [4, 4, 4], 'starts': [125, 126, 127],
                              'commits': [4, 4, 4], 'prefix_tokens': [p['prompt_token_ids'] for p in prompts],
                              'query_tokens': [[row['token'] for row in values] for values in rows],
                              'serving_rows': [row for values in rows for row in values]})
    result = {'layers': args.layers, 'plan': plan, 'pass': False, 'status': 'RUNNING',
              'precision_gate_deferred': args.defer_precision_gate,
              'scope': 'real weights; target-only; not full model quality or throughput'}
    try:
        result['ranks'] = llm.collective_rpc('packed_target_probe', args=(plan,))
        result['target_pass'] = len(result['ranks']) == 8 and all(rank['pass'] for rank in result['ranks'])
        if not result['target_pass']:
            raise RuntimeError('Packed target all-query or continuation gate failed')
        if args.scheduled_serving:
            from tools.validation.executor.packed_serving_check import run
            result['scheduled_serving'] = run(llm, args.out)
        if args.static_replay:
            result['static_replay'] = llm.collective_rpc('packed_static_replay_probe')
            if len(result['static_replay']) != 8 or not all(rank['pass'] for rank in result['static_replay']):
                raise RuntimeError('Packed same-address recorded replay failed')
        if args.advancing_replay:
            result['advancing_replay'] = llm.collective_rpc('packed_advancing_replay_probe', args=(args.native_packed_swa,))
            if len(result['advancing_replay']) != 8 or not all(rank['pass'] for rank in result['advancing_replay']):
                raise RuntimeError('Packed advancing recorded replay failed')
        if args.target_features:
            layers = (0, 1) if args.layers == 2 else (0, 15, 31, 47, 69)
            result['target_features'] = llm.collective_rpc('packed_feature_probe', args=(layers, args.native_packed_swa))
            if len(result['target_features']) != 8 or not all(rank['pass'] for rank in result['target_features']):
                raise RuntimeError('Target auxiliary feature boundary audit failed')
        if args.dflash_cycle:
            sweep = args.cycle_batch_sweep or args.cycle_row_sweep or args.cycle_binding_sweep or args.cycle_gqa_sweep or args.cycle_moe_sweep
            result['dflash_sweep'] = []
            for batch in args.cycle_batch_sweep or (len(prompts),):
                for extent, binding_mode, gqa, moe_index, moe_mode in ((n, mode, g, i, m) for n in args.cycle_row_sweep or (args.cycle_verify_rows,)
                                                  for mode in args.cycle_binding_sweep or ('static',)
                                                  for g in args.cycle_gqa_sweep or ('broadcast',)
                                                  for i, m in enumerate(args.cycle_moe_sweep or ('production',))):
                    cycle_plan = {'prompt_ids': [p['prompt_token_ids'] for p in prompts[:batch]],
                                  'draft_checkpoint': str(Path(args.model)/'dflash'), 'native_swa': args.native_packed_swa,
                                  'cycles': args.cycle_steps, 'verify_rows': extent,
                                  'record_cycle': args.record_dflash_cycle, 'routes': args.cycle_routes,
                                  'static_pages': binding_mode == 'static',
                                  'profile': args.profile_dflash_cycle,
                                  'fold_gqa': gqa == 'folded',
                                  'compact_row_diagnostic': moe_mode == 'compact12',
                                  'label': (f'dflash-b{batch}-t{extent}' +
                                            (f'-{binding_mode}' if args.cycle_binding_sweep else '')+
                                            (f'-gqa-{gqa}' if args.cycle_gqa_sweep else '')+
                                            (f'-moe{moe_index}-{moe_mode}' if args.cycle_moe_sweep else '')) if sweep else 'dflash-target-cycle'}
                    ranks = llm.collective_rpc('dflash_cycle_probe', args=(cycle_plan,))
                    emitted = [[c['emitted_ids'] for c in rank['cycles']] for rank in ranks]
                    equal = len(ranks) == 8 and all(values == emitted[0] for values in emitted)
                    entry = dict(batch=batch, verify_rows=extent, binding_mode=binding_mode, gqa_mode=gqa, moe_mode=moe_mode,
                                 ranks=ranks, replicas_equal=equal)
                    routes_equal = True
                    if args.cycle_routes:
                        routes = [[c['route_distribution'] for c in rank['cycles']] for rank in ranks]
                        routes_equal = len(ranks) == 8 and all(values == routes[0] for values in routes)
                        entry['route_replicas_equal'] = routes_equal
                    result['dflash_sweep'].append(entry)
                    if not sweep:
                        result.update(dflash_cycles=ranks, dflash_replicas_equal=equal)
                    # Persist each case before the next RPC. A later case's
                    # failure cannot erase already completed private evidence.
                    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
                    if len(ranks) != 8 or not all(r['pass'] for r in ranks) or not equal or not routes_equal:
                        raise RuntimeError('Real DFlash/target functional or TP-replica check failed')
        result.update({'pass': True, 'status': 'COMPLETE'})
    except BaseException as error:
        result.update({'status': 'FAILED', 'error': repr(error), 'pass': False})
        raise
    finally:
        (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
