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
    p.add_argument('--serving-teacher', action='store_true',
                   help='Also compare packed multi-position logits against ordinary vLLM serving')
    p.add_argument('--scheduled-serving', action='store_true',
                   help='Also run the experimental compact consumer of actual scheduling and allocator KV')
    p.add_argument('--static-replay', action='store_true',
                   help='Probe same-address recorded target replay; not explicit graph construction or throughput')
    p.add_argument('--advancing-replay', action='store_true',
                   help='Update recorded token/position/KV bindings across accepted/rejected input prefixes')
    args = p.parse_args()
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
    plan = {'cases': [{'name': 'ragged-small', 'rows': [1, 4, 8], 'starts': [127, 126, 0], 'commits': [1, 2, 8]}]}
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
            result['advancing_replay'] = llm.collective_rpc('packed_advancing_replay_probe')
            if len(result['advancing_replay']) != 8 or not all(rank['pass'] for rank in result['advancing_replay']):
                raise RuntimeError('Packed advancing recorded replay failed')
        result.update({'pass': True, 'status': 'COMPLETE'})
    except BaseException as error:
        result.update({'status': 'FAILED', 'error': repr(error), 'pass': False})
        raise
    finally:
        (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
