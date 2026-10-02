"""Measure tokens delivered by vLLM, including scheduler/transport/worker time.

Arrival groups stay intact: speculative tokens delivered together are never
assigned invented per-token timestamps. This is an engine-client measurement,
not an HTTP latency measurement and not accelerator-only timing.
"""
import time


def summarize(events, request_ids, *, warmup_tokens=4):
    per_request = []
    for request in request_ids:
        rows = [e for e in events if e['request_id'] == request]
        if not rows or not rows[-1]['finished']:
            raise ValueError('Missing completed request: ' + request)
        cutoff = next((e for e in rows if e['total_tokens'] >= warmup_tokens), None)
        end = rows[-1]
        count = end['total_tokens'] - cutoff['total_tokens'] if cutoff else 0
        span = end['arrival_ns'] - cutoff['arrival_ns'] if cutoff else 0
        per_request.append(dict(request_id=request, total_tokens=end['total_tokens'],
            measured_tokens=count, decode_span_ns=span,
            tps=count*1e9/span if count > 0 and span > 0 else None,
            first_token_ns=rows[0]['arrival_ns'], finished_ns=end['arrival_ns'],
            cutoff_ns=cutoff['arrival_ns'] if cutoff else None,
            arrival_groups=len(rows)))
    aggregate = None
    if all(r['cutoff_ns'] is not None for r in per_request):
        begin = max(r['cutoff_ns'] for r in per_request)
        end = max(r['finished_ns'] for r in per_request)
        count = sum(e['new_tokens'] for e in events if e['arrival_ns'] > begin)
        concurrent=[]
        for row in per_request:
            emitted=sum(e['new_tokens'] for e in events if
                        e['request_id']==row['request_id'] and e['arrival_ns']>begin)
            span=row['finished_ns']-begin
            concurrent.append(dict(request_id=row['request_id'],tokens=emitted,span_ns=max(0,span),
                tps=emitted*1e9/span if emitted and span>0 else None))
        aggregate = dict(begin_ns=begin, end_ns=end, tokens=count,
            tps=count*1e9/(end-begin) if count and end > begin else None,
            per_request=concurrent,
            scope='common post-warmup window, including request completion tails')
    return dict(per_request=per_request, aggregate=aggregate, warmup_tokens=warmup_tokens,
        scope='actual cumulative engine-client arrivals; prefill/first tokens reported separately; '
              'capture and page transitions inside the measured window remain included')


def run(engine, prompts, params, *, tag, warmup_tokens=4, clock=time.perf_counter_ns):
    """Use the engine's add_request/step API, with CUMULATIVE output enabled.

    Caller supplies sampling parameters; LLM.generate/enqueue would overwrite
    output_kind to FINAL_ONLY in the qualified vLLM source. No worker hooks or
    timing RPCs occur in this loop. Any owned requests are aborted on failure.
    """
    if engine.has_unfinished_requests():
        raise ValueError('Arrival measurement requires an empty request queue')
    request_ids = [f'{tag}-{i}' for i in range(len(prompts))]
    events, final, previous, internal_ids = [], {}, {}, []
    begin = clock()
    try:
        for rid, prompt in zip(request_ids, prompts):
            internal_ids.append(engine.add_request(rid, prompt, params.clone()))
        step = 0
        while engine.has_unfinished_requests():
            outputs = engine.step()
            arrival = clock() - begin
            step += 1
            for out in outputs:
                rid = out.request_id
                if rid not in request_ids or len(out.outputs) != 1 or rid in final:
                    raise ValueError('Unexpected or already finished request output')
                values = list(out.outputs[0].token_ids)
                old = previous.get(rid, [])
                if values[:len(old)] != old or len(values) < len(old):
                    raise ValueError('Expected monotonically growing cumulative token IDs')
                new = len(values) - len(old)
                if new or out.finished:
                    events.append(dict(request_id=rid, step=step, arrival_ns=arrival,
                        new_tokens=new, total_tokens=len(values), finished=out.finished))
                previous[rid] = values
                if out.finished:
                    final[rid] = out
        summary = summarize(events, request_ids, warmup_tokens=warmup_tokens)
        return [final[rid] for rid in request_ids], dict(events=events, summary=summary,
            wall_ns=clock()-begin, steps=step, request_ids=request_ids)
    except BaseException:
        if internal_ids:
            engine.abort_request(internal_ids, internal=True)
        raise
