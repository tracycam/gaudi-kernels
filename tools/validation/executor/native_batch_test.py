"""Same-process bridge/native multi-request comparison, including KV page edges."""
import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layers', type=int, choices=(2,70), default=2)
    p.add_argument('--batch-sizes', default='2,3')
    p.add_argument('--tokens', type=int, default=16)
    p.add_argument('--arrivals',action='store_true',
                   help='Measure actual per-request cumulative engine-client token arrivals')
    p.add_argument('--prompts-jsonl',type=Path,
                   help='Pinned local corpus with original prompt_token_ids and their SHA256')
    p.add_argument('--timing-control',type=Path,
                   help='Native/native policy ABBA; requires the same document as --teacher-control')
    p.add_argument('--teacher-control',type=Path,
                   help='Optional typed control selection; compares every request/query, outside timing')
    a = p.parse_args()
    from gaudi_kernels.engine.context import context
    from gaudi_kernels.serving.host_placement import vllm_kwargs
    from vllm import LLM, SamplingParams
    if a.layers != context().startup.layers or not 8 <= a.tokens <= 160:
        raise ValueError('Explicit model scope and bounded token count required')
    sizes = [int(v) for v in a.batch_sizes.split(',')]
    if any(b not in context().selection.engine.runtime.buckets.batch or not 1 < b <= 8 for b in sizes):
        raise ValueError('Batch size outside manifest admission')
    candidate=context().selection.to_dict()
    timing_control=None
    if a.timing_control is not None:
        timing_control=json.loads(a.timing_control.read_text())
        if a.teacher_control is None or timing_control!=json.loads(a.teacher_control.read_text()):
            raise ValueError('Native policy timing requires its actual teacher-forced control')
    opts = dict(model='/data/models/MiMo-V2.6-Pro-RL', trust_remote_code=True,
        tensor_parallel_size=8, enable_expert_parallel=False, dtype='bfloat16',
        enforce_eager=False, async_scheduling=False, max_model_len=8192, block_size=128,
        max_num_seqs=max(sizes), gpu_memory_utilization=.3 if a.layers==2 else .95,
        max_num_batched_tokens=512, disable_log_stats=True, enable_prefix_caching=False,
        seed=0, limit_mm_per_prompt={'audio':0,'image':0,'video':0})
    opts.update(vllm_kwargs())
    if a.layers == 2:
        cfg = json.loads((Path(opts['model'])/'config.json').read_text())
        opts['hf_overrides'] = {'num_hidden_layers':2,
            'hybrid_layer_pattern':cfg['hybrid_layer_pattern'][:2], 'moe_layer_freq':cfg['moe_layer_freq'][:2]}
        opts['num_gpu_blocks_override'] = 256
    result = {'status':'RUNNING', 'layers':a.layers, 'candidate_accepted':False,
        'scope':'native batch execution equivalence; model quality/performance qualification is separate',
        'config':opts, 'runs':[]}
    if timing_control is not None:
        result.update(scope='native/native arithmetic-policy ABBA after all-request teacher gate',
                      timing_control=timing_control,candidate=candidate)
    corpus_prompts=None
    if a.prompts_jsonl is not None:
        from tools.validation.executor.batch_prompts import load
        corpus_prompts,result['corpus']=load(a.prompts_jsonl,needed=max(sizes),
                                             maximum_length=opts['max_model_len']-a.tokens)
    def save():
        tmp = a.out/'result.json.tmp'; tmp.write_text(json.dumps(result,indent=2)+'\n')
        tmp.replace(a.out/'result.json')
    save()
    llm = LLM(**opts)
    params = SamplingParams(temperature=0, max_tokens=a.tokens, ignore_eos=True)
    if a.arrivals:
        from vllm.sampling_params import RequestOutputKind
        params.output_kind=RequestOutputKind.CUMULATIVE
    token_ids = llm.get_tokenizer().encode('The quick brown fox jumps over the lazy dog. ')
    try:
        if context().selection.engine.runtime.native_enabled:
            # Verify serving startup before any benchmark native_configure RPC.
            startup_prompts=[{'prompt_token_ids':(token_ids*128)[:125+i]} for i in range(sizes[0])]
            outputs=llm.generate(startup_prompts,params.clone(),use_tqdm=False)
            ranks=llm.collective_rpc('native_summary')
            if any(not any(s.get('kind')=='batch' for s in r['native_steps']) for r in ranks):
                raise RuntimeError('Serving startup did not activate native replay on every rank')
            result['startup_native']={'pass':True,'ranks':ranks,
                'ids':[o.outputs[0].token_ids for o in outputs],
                'scope':'ordinary LLM request before any benchmark native_configure RPC'}
            save()
        if a.teacher_control is not None:
            from tools.validation.executor.batch_quality import compare
            control=json.loads(a.teacher_control.read_text())
            result['teacher_forced']=[]
            for batch in sizes:
                result['teacher_forced'].append(compare(llm,batch,token_ids,control,candidate,a.out))
                save()
            llm.collective_rpc('production_policy',args=(candidate,))
        for batch in sizes:
            # Stagger lengths across 128-token boundaries and finish requests together.
            lengths = [125+i for i in range(batch)]
            prompts = [{'prompt_token_ids':(token_ids*128)[:n]} for n in lengths]
            if corpus_prompts is not None:
                prompts=corpus_prompts[:batch]
                lengths=[len(p['prompt_token_ids']) for p in prompts]
            control = None
            for arm, active in (('A1',False),('B1',True),('B2',True),('A2',False)):
                policy='candidate'
                if timing_control is not None:
                    llm.collective_rpc('native_configure',args=(False,False,False,'compact'))
                    policy='candidate' if active else 'control'
                    llm.collective_rpc('production_policy',args=(candidate if active else timing_control,))
                    active=True
                llm.collective_rpc('native_configure', args=(active,True,True,'compact'))
                start = time.perf_counter_ns()
                arrivals=None
                if a.arrivals:
                    from tools.validation.executor.token_arrivals import run
                    outputs,arrivals=run(llm.llm_engine,prompts,params,tag=f'b{batch}-{arm}')
                else:
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                elapsed = time.perf_counter_ns()-start
                ids = [o.outputs[0].token_ids for o in outputs]
                if control is None:
                    control = ids
                ranks = llm.collective_rpc('native_summary')
                row = {'batch':batch, 'arm':arm, 'active':active, 'wall_ns':elapsed,
                    'prompt_lengths':lengths, 'ids':ids, 'matches_bridge':ids==control, 'ranks':ranks,
                    'token_arrivals':arrivals,
                    'policy':policy,'comparison':'native_policy_abba' if timing_control is not None else 'bridge_native',
                    'outputs':[dict(request_id=o.request_id,text=o.outputs[0].text,
                        finish_reason=o.outputs[0].finish_reason,stop_reason=o.outputs[0].stop_reason)
                        for o in outputs]}
                result['runs'].append(row); save()
                if ids != control and timing_control is None:
                    raise RuntimeError('Native batch output differs from same-shape bridge')
                if active and any(not any(s.get('kind')=='batch' for s in r['native_steps']) for r in ranks):
                    raise RuntimeError('No actual batch native replay on every TP rank')
        result['status'] = 'NATIVE_POLICY_ABBA_COMPLETE' if timing_control is not None else 'EXECUTION_EQUIVALENCE_PASS'
    except BaseException as exc:
        result.update(status='FAILED', error=repr(exc))
        raise
    finally:
        try:
            llm.collective_rpc('native_configure', args=(False,False,False,'compact'))
        except Exception as exc:
            result['cleanup_error'] = repr(exc)
        save()


if __name__ == '__main__':
    main()
