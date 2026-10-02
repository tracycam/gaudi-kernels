"""Frozen benchmark labels to explicit capture-row coverage; no serving import."""

def batch_capture_gate(policy, rows, states):
    suffixes=set(policy.split('+')[1:])
    wanted=list(range(1,9)) if 'moe_compact8' in suffixes else [1]
    buckets=[state.get('serving_decode_batch_buckets') for state in states]
    if any(b is not None for b in buckets):
        if not buckets or any(not b or b!=buckets[0] for b in buckets):
            return {'pass':False,'reason':'Inconsistent decode bucket coverage'}
        options=[v for v in buckets[0] if type(v) is int and v>=rows]
        if not options:return {'pass':False,'reason':'Missing static capture bucket'}
        rows=min(options)
    mode='compact' if rows in wanted else 'broadcast'
    passed=len(states)==8 and all(
        state.get('moe_dispatch',{}).get('compact_rows')==wanted
        and state.get('moe_dispatch',{}).get('python_capture_calls_by_rows',{}).get(str(rows),{}).get(mode,0)>0
        for state in states)
    return {'pass':passed,'rows':rows,'expected_mode':mode,'expected_compact_rows':wanted,
            'scope':'All-rank capture metadata, not physical invocation counts'}
