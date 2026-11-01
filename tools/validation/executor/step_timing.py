"""Complete ordinary engine-step timing; no execution policy or model hooks."""
import time

def install_step_timer(engine, steps, counts, *, clock=time.perf_counter_ns):
    """Own the engine callable separately from mutable model/audit locals."""
    engine_step=engine.step
    def timed_step():
        begin=clock();outputs=engine_step();end=clock();delta=0;per_request=[]
        for output in outputs:
            if not output.outputs:continue
            count=len(output.outputs[0].token_ids);increment=max(0,count-counts.get(output.request_id,0));delta+=increment;counts[output.request_id]=count
            if increment:per_request.append(increment)
        steps.append(dict(begin_ns=begin,end_ns=end,ms=(end-begin)/1e6,new_tokens=delta,emitted=max(counts.values(),default=0),
                          emitted_min=min(counts.values(),default=0),known_requests=len(counts),per_request_new=per_request))
        return outputs
    engine.step=timed_step

