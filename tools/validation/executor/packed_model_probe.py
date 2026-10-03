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
    args = p.parse_args()
    from gaudi_kernels.engine.context import context
    from gaudi_kernels.serving.host_placement import vllm_kwargs
    from vllm import LLM
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
    result = {'layers': args.layers, 'plan': plan, 'scope': 'real weights; target-only; not full model quality or throughput'}
    try:
        result['ranks'] = llm.collective_rpc('packed_target_probe', args=(plan,))
        result['pass'] = len(result['ranks']) == 8 and all(rank['pass'] for rank in result['ranks'])
        if not result['pass']:
            raise RuntimeError('Packed target all-query or continuation gate failed')
    finally:
        (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
