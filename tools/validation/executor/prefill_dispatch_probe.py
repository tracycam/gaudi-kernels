"""Resident production MoE ABBA and chunked-prefill validation through vLLM.

Whole engine-client arrival intervals are measured. Teacher logits and memory
queries are collected outside timing. Partial-model runs are never accepted.
"""
import argparse
import copy
import json
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--layers',type=int,choices=(2,70),default=2)
    p.add_argument('--prompt-tokens',type=int,default=4097)
    p.add_argument('--output-tokens',type=int,default=32)
    p.add_argument('--arms',type=int,choices=(2,4),default=4)
    a=p.parse_args()
    from gaudi_kernels.engine.context import context
    from gaudi_kernels.serving.host_placement import vllm_kwargs
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from tools.validation.executor.token_arrivals import run as arrivals
    from tools.validation.executor.batch_quality import generate_with_ids,attach_owners,compare_rows
    ctx=context();chunk=ctx.selection.engine.runtime.prefill_chunk_tokens
    if a.layers!=ctx.startup.layers or not 16<=a.output_tokens<=128 or not chunk<a.prompt_tokens<=32768:
        raise ValueError('Explicit layer scope, bounded output and multiple prefill chunks required')
    candidate=ctx.selection.to_dict();control=copy.deepcopy(candidate)
    control['engine']['decode']['moe']['dispatch']['grouped_rows']=[]
    if chunk not in candidate['engine']['decode']['moe']['dispatch']['grouped_rows']:
        raise ValueError('Candidate must actually select the measured prefill chunk')
    opts=dict(model='/data/models/MiMo-V2.6-Pro-RL',trust_remote_code=True,
              tensor_parallel_size=8,enable_expert_parallel=False,dtype='bfloat16',
              enforce_eager=False,async_scheduling=False,max_model_len=a.prompt_tokens+a.output_tokens+128,
              block_size=128,max_num_seqs=1,gpu_memory_utilization=.3 if a.layers==2 else .95,
              disable_log_stats=True,enable_prefix_caching=False,seed=0,
              limit_mm_per_prompt={'audio':0,'image':0,'video':0})
    opts.update(vllm_kwargs())
    if a.layers==2:
        cfg=json.loads((Path(opts['model'])/'config.json').read_text())
        opts.update(hf_overrides={'num_hidden_layers':2,'hybrid_layer_pattern':cfg['hybrid_layer_pattern'][:2],
                                 'moe_layer_freq':cfg['moe_layer_freq'][:2]},num_gpu_blocks_override=512)
    result=dict(status='RUNNING',layers=a.layers,chunk=chunk,config=opts,control=control,candidate=candidate,
                runs=[],teacher={},model_candidate_accepted=False,
                scope='production model entry; client TTFT includes all scheduler/transfer/capture work')
    def save():
        temporary=a.out/'result.json.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(a.out/'result.json')
    save();llm=LLM(**opts);rpc=llm.collective_rpc
    tokenizer=llm.get_tokenizer()
    # Deterministic real repository prose. Persist exact IDs for cross-chunk runs.
    source=(Path(__file__).resolve().parents[3]/'README.md').read_text()
    body=tokenizer.encode(source)
    question=tokenizer.encode('\nVerification code: 7391. Output exactly 7391 first, then explain briefly.\n',add_special_tokens=False)
    template=tokenizer.apply_chat_template([{'role':'user','content':'__GK_CONTENT__'}],
                                          tokenize=False,add_generation_prompt=True,enable_thinking=False)
    left,right=template.split('__GK_CONTENT__')
    prefix=tokenizer.encode(left,add_special_tokens=False)
    suffix=question+tokenizer.encode(right,add_special_tokens=False)
    filler=(body*((a.prompt_tokens+len(body)-1)//len(body)))[:a.prompt_tokens-len(prefix)-len(suffix)]
    prompt=prefix+filler+suffix
    assert len(prompt)==a.prompt_tokens
    prompts=[{'prompt_token_ids':prompt}]
    result['prompt_token_ids']=prompt;save()
    params=SamplingParams(temperature=0,max_tokens=a.output_tokens,ignore_eos=True)
    params.output_kind=RequestOutputKind.CUMULATIVE
    def policy(document):
        rpc('native_configure',args=(False,False,False,'compact'))
        changed=rpc('production_policy',args=(document,))
        if len(changed)!=8 or any(r['details']['selected_layer_count']!=a.layers for r in changed):
            raise RuntimeError('Incomplete model policy installation')
        return changed
    try:
        # Teacher-forced same prompt and continuation. Only capture decode
        # logits, which include the complete preceding prefill computation.
        policy(control)
        anchor=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=1),use_tqdm=False)[0].outputs[0].token_ids[0]
        forced=SamplingParams(temperature=0,max_tokens=8,ignore_eos=True,allowed_token_ids=[anchor])
        records={}
        for name,doc in (('control',control),('candidate',candidate)):
            policy(doc)
            rpc('production_quality_capture',args=(f'prefill-{name}','batch',1))
            outputs,mapping=generate_with_ids(llm,prompts,forced)
            captured=rpc('production_quality_capture',args=(None,))
            records[name]=attach_owners(next(r['captured_forwards'] for r in captured if r['rank']==0),outputs,mapping)
        checks=compare_rows(records['control'],records['candidate'])
        passed=len(checks)>=4 and all(r['check']['pass'] for r in checks)
        result['teacher']={'pass':passed,'checks':checks,'records':records,'anchor':anchor}
        save()
        if not passed:raise RuntimeError('Teacher-forced prefill-conditioned logits failed')
        for arm,name in enumerate(('control','candidate','candidate','control')[:a.arms]):
            policy(control if name=='control' else candidate)
            # Complete identical warmup outside the reported interval.
            for trial in ('warmup','measured'):
                rpc('device_memory_snapshot',args=(True,))
                outputs,timing=arrivals(llm.llm_engine,prompts,params,tag=f'p{arm}-{trial}')
                emitted=outputs[0].outputs[0]
                state=rpc('production_snapshot')
                memory=rpc('device_memory_snapshot')
                t=timing['summary']['per_request'][0]
                row=dict(arm=arm,policy=name,trial=trial,timing=timing,
                         prefill_completion_tps=len(prompt)*1e9/t['first_token_ns'],
                         ttft_ms=t['first_token_ns']/1e6,ids=list(emitted.token_ids),text=emitted.text,
                         token_count_pass=len(emitted.token_ids)==a.output_tokens,
                         verification_code_present='7391' in emitted.text,
                         memory=memory,production_state=state)
                result['runs'].append(row);save()
                if not row['token_count_pass']:raise RuntimeError('Incomplete generation')
        measured=[r for r in result['runs'] if r['trial']=='measured']
        candidate_runs=[r for r in measured if r['policy']=='candidate']
        control_runs=[r for r in measured if r['policy']=='control']
        actual=bool(candidate_runs) and all(
            len(row['production_state'])==8 and all(
                rank['moe_dispatch']['python_capture_calls_by_rows'].get(str(chunk),{}).get('grouped',0)>=a.layers-1
                for rank in row['production_state']) for row in candidate_runs)
        quality=all(r['verification_code_present'] for r in measured)
        ttft={name:statistics.median(r['ttft_ms'] for r in measured if r['policy']==name)
              for name in ('control','candidate')}
        decode={name:statistics.median(r['timing']['summary']['per_request'][0]['tps']
                                      for r in measured if r['policy']==name)
                for name in ('control','candidate')}
        performance=ttft['candidate']<=ttft['control']*1.05 and decode['candidate']>=decode['control']*.95
        result.update(status='COMPLETE',grouped_on_all_ranks=actual,quality_pass=quality,
                      performance_gate_pass=performance,median_ttft_ms=ttft,median_decode_tps=decode,
                      model_candidate_accepted=a.layers==70 and a.arms==4 and passed and actual and quality and performance)
        save()
    except BaseException as error:
        result.update(status='FAILED',error=repr(error));save();raise
    finally:
        del llm


if __name__=='__main__':main()
