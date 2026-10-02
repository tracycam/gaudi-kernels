"""Require actual simultaneous decoding; never infer it from requested batch."""
import statistics


def full_batch_window(steps, batch, token_limit):
    windows=[];current=[]
    for step in steps:
        valid=(step.get('known_requests')==batch and step.get('emitted_min',0)>=32
               and step['emitted']<token_limit and step.get('per_request_new')==[1]*batch)
        if valid:current.append(step)
        else:
            if current:windows.append(current)
            current=[]
    if current:windows.append(current)
    window=max(windows,key=len,default=[])
    if len(window)<16:
        return {'qualified':False,'reason':'fewer than 16 consecutive steps with every request emitting one token',
                'longest_full_batch_steps':len(window)}
    seconds=(window[-1]['end_ns']-window[0]['begin_ns'])/1e9
    return {'qualified':True,'actual_batch':batch,'steps':len(window),
            'aggregate_decode_tps':batch*len(window)/seconds,
            'median_step_ms':statistics.median(s['ms'] for s in window),
            'first_emitted_min':window[0]['emitted_min'],'last_emitted_max':window[-1]['emitted'],
            'scope':'longest contiguous matured full-batch window; no gaps or slow steps removed within it'}
