"""Matched long-context serving check using the already loaded checkpoint."""
import statistics,time
from tools.validation.executor.serving_gate import performance_qualified

def check_long_context(llm,context,tokens,layers,single_rpc,reuse_pages,result,steps,counts,save):
    from vllm import SamplingParams
    tok=llm.get_tokenizer()
    prefix=tok.encode('Record: the verification code for the copper lantern is 7391.\n',add_special_tokens=False)
    filler=tok.encode('This document describes reproducible experiments and laboratory observations.\n',add_special_tokens=False)
    suffix=tok.encode('\nQuestion: What is the verification code for the copper lantern? State the code and explain the evidence.\nAnswer:',add_special_tokens=False)
    n=context-len(prefix)-len(suffix);assert n>0
    prompt={'prompt_token_ids':prefix+(filler*((n+len(filler)-1)//len(filler)))[:n]+suffix}
    result.update(context=context,tokens=tokens,scope='long-context full-attention KV and128-token SWA boundary; B1greedy')
    reference=None
    for name,active in [('bridge',False),('native',True)]:
        llm.collective_rpc('native_configure',args=(active,active and single_rpc,active and reuse_pages))
        steps.clear();counts.clear();begin=time.monotonic()
        out=llm.generate([prompt],SamplingParams(temperature=0,max_tokens=tokens,ignore_eos=True),use_tqdm=False)[0].outputs[0]
        ids=list(out.token_ids)
        if reference is None:reference=ids
        raw=llm.collective_rpc('native_summary')
        run=dict(name=name,active=active,ids=ids,text=out.text,wall_s=time.monotonic()-begin,steps=list(steps),ranks=raw,
                 quality_pass='7391' in out.text,bitwise_token_match=ids==reference,token_count_pass=len(ids)==tokens)
        run['native_used_all_ranks']=len(raw)==8 and all(x['native_steps'] for x in raw)
        run['all_capture_gates_pass']=all(c['status']=='NUMERICAL_REPLAY_PASS' for x in raw for c in x['captures'])
        run['performance_qualified']=performance_qualified(layers,run)
        steady=[x for x in steps if x['new_tokens']>0 and x['emitted']>=min(32,tokens//2) and x['emitted']<tokens]
        if steady:
            key='serving_timing' if run['performance_qualified'] else 'diagnostic_timing'
            run[key]=dict(tps=sum(x['new_tokens'] for x in steady)/((steady[-1]['end_ns']-steady[0]['begin_ns'])/1e9),
                          steady_steps=len(steady),p50_itl_ms=statistics.median(x['ms'] for x in steady),
                          scope='coordinator wall; includes all boundary/capture costs in this window')
        result['runs'].append(run);save()
        assert run['bitwise_token_match'] and run['token_count_pass'] and run['all_capture_gates_pass']
        if active:assert run['native_used_all_ranks']
        if layers==70:assert run['quality_pass']
    result['status']='PASS' if layers==70 else 'DIAGNOSTIC';save()
