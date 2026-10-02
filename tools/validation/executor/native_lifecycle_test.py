"""Bounded stop-token, cancellation and new-request checks on one loaded model."""
def check_lifecycle(llm,prompt,baseline,single_rpc,reuse_pages):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    # Exercise the scheduler's stop-token handling with an actual model token.
    # This is not a claim that the prompt naturally reaches model EOS.
    index=next(i for i,x in enumerate(baseline) if i>=8 and x not in baseline[:i])
    stop_id=baseline[index];records=[];stop_reference=None
    for active in (False,True):
        llm.collective_rpc('native_configure',args=(active,active and single_rpc,active and reuse_pages))
        out=llm.generate([prompt],SamplingParams(temperature=0,max_tokens=len(baseline),
            stop_token_ids=[stop_id],ignore_eos=True),use_tqdm=False)[0].outputs[0]
        ids=list(out.token_ids)
        if stop_reference is None:stop_reference=ids
        assert out.finish_reason=='stop' and ids==stop_reference and ids==baseline[:len(ids)]
        raw=llm.collective_rpc('native_summary')
        if active:assert all(x['native_steps'] for x in raw),'Stop-token check did not use native steps'
        records.append(dict(active=active,operation='stop_token',stop_id=stop_id,
                            ids=ids,finish_reason=out.finish_reason,ranks=raw))
    # Do not reset or reconfigure between requests: test automatic invalidation.
    engine=llm.llm_engine;rid='native-cancellation-probe'
    engine.add_request(rid,prompt,SamplingParams(temperature=0,max_tokens=len(baseline),ignore_eos=True,
                                              output_kind=RequestOutputKind.CUMULATIVE))
    observed=[]
    while engine.has_unfinished_requests() and len(observed)<12:
        for out in engine.step():
            if out.outputs:observed=list(out.outputs[0].token_ids)
    assert observed==baseline[:len(observed)] and len(observed)>=12
    engine.abort_request([rid])
    raw=llm.collective_rpc('native_summary')  # ordered after the abort message
    assert not engine.has_unfinished_requests()
    records.append(dict(operation='cancel',ids=observed,ranks=raw))
    fresh=llm.generate([prompt],SamplingParams(temperature=0,max_tokens=16,ignore_eos=True),use_tqdm=False)[0].outputs[0]
    assert list(fresh.token_ids)==baseline[:16]
    raw=llm.collective_rpc('native_summary')
    assert all(len(x['captures'])>=3 and x['native_steps'] for x in raw)
    records.append(dict(operation='new_request_after_cancel',ids=list(fresh.token_ids),ranks=raw))
    return dict(status='PASS',scope='explicit stop token, cancel and next request; not natural EOS or graph eviction',records=records)
