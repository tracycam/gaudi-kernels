"""Same ordinary model program for upstream/fork wheels; no executor-control RPCs."""
import argparse
import json
from pathlib import Path
import statistics
import time

from gaudi_kernels.engine.context import context
from tools.validation.executor.native_service_test import install_step_timer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--batch-sizes', default='1,2')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--diagnose', action='store_true',
                        help='After all timing runs, collect bounded nested host-call timings')
    args = parser.parse_args()
    runtime = context()
    if runtime.selection.engine.runtime.executor != 'pytorch':
        raise ValueError('This baseline must not use an external execution engine')
    batches = tuple(int(value) for value in args.batch_sizes.split(','))
    if not batches or any(value < 1 or value > 8 for value in batches):
        parser.error('batch sizes must be 1..8')
    if not 16 <= args.tokens <= 512 or not 1 <= args.rounds <= 4:
        parser.error('bounded tokens/rounds required')
    from vllm import LLM, SamplingParams
    from gaudi_kernels.serving.host_placement import vllm_kwargs
    options = dict(model='/data/models/MiMo-V2.6-Pro-RL', trust_remote_code=True,
                   tensor_parallel_size=8, enable_expert_parallel=False, dtype='bfloat16',
                   enforce_eager=False, async_scheduling=False, block_size=128,
                   max_model_len=max(8192, args.context+args.tokens+128),
                   max_num_seqs=max(batches), gpu_memory_utilization=.95,
                   disable_log_stats=True, enable_prefix_caching=False, seed=0,
                   limit_mm_per_prompt={'audio': 0, 'image': 0, 'video': 0})
    options.update(vllm_kwargs())
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
    token = llm.get_tokenizer().encode(' Describe software architecture.', add_special_tokens=False)
    prompts = [{'prompt_token_ids': (token*((args.context+len(token)-1)//len(token)))[:args.context]}
               for _ in range(max(batches))]
    sample = SamplingParams(temperature=0, max_tokens=args.tokens, ignore_eos=True)
    for batch in batches:
        # Compile/warm separately from measured real requests.
        llm.generate(prompts[:batch], SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
                     use_tqdm=False)
        for repeat in range(args.rounds):
            counts.clear()
            steps.clear()
            begin = time.perf_counter_ns()
            outputs = llm.generate(prompts[:batch], sample, use_tqdm=False)
            end = time.perf_counter_ns()
            rows = [dict(ids=list(output.outputs[0].token_ids), text=output.outputs[0].text)
                    for output in outputs]
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
            result['runs'].append(dict(batch=batch, repeat=repeat, outputs=rows, steps=list(steps),
                total_tokens=sum(len(row['ids']) for row in rows), wall_ms=(end-begin)/1e6,
                total_tokens_per_second=sum(len(row['ids']) for row in rows)*1e9/(end-begin),
                resident_step_count=len(resident),
                resident_tokens_per_second=sum(step['new_tokens'] for step in resident)*1000/
                                           sum(step['ms'] for step in resident),
                resident_p50_ms=statistics.median(step['ms'] for step in resident)))
            save()
    if args.diagnose:
        # Run only after measured requests. No new executor, device sync or
        # tensor read is inserted by these observers. Restore before snapshot.
        llm.collective_rpc('framework_timing', args=(True,))
        try:
            llm.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=64, ignore_eos=True),
                         use_tqdm=False)
        finally:
            result['host_timings'] = llm.collective_rpc('framework_timing', args=(False,))
            save()
    result['ranks'] = llm.collective_rpc('framework_execution_snapshot')
    if any(rank['external_executors'] for rank in result['ranks']):
        raise AssertionError('External execution engine loaded')
    result['status'] = 'PASS'
    save()
    print(json.dumps(dict(status=result['status'], layers=result['model_layers'],
                          runs=[{key:run[key] for key in ('batch','repeat','resident_tokens_per_second',
                                                         'total_tokens_per_second')}
                                for run in result['runs']]), indent=2), flush=True)


if __name__ == '__main__':
    main()
