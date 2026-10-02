"""Acceptance and actual-route accounting, independent of the serving backend."""
from collections import Counter

def acceptance(steps):
    rows=[r for step in steps for r in step['requests']]
    if not rows:return dict(request_cycles=0,tau=None,conditional_alpha=[])
    for r in rows:
        k=len(r['drafts']);n=len(r['output_ids']);a=r['accepted_drafts']
        if not (r['emitted']==n==a+1 and 0<=a<=k and r['output_ids'][:-1]==r['drafts'][:a]):
            raise ValueError('invalid acceptance record')
    maximum=max(len(r['drafts'])for r in rows);positions=[]
    for j in range(maximum):
        eligible=[r for r in rows if len(r['drafts'])>j and r['accepted_drafts']>=j]
        passed=sum(r['accepted_drafts']>j for r in eligible)
        positions.append(dict(position=j+1,reached=len(eligible),accepted=passed,alpha=passed/len(eligible)if eligible else None))
    return dict(request_cycles=len(rows),emitted=sum(r['emitted']for r in rows),
        tau=sum(r['emitted']for r in rows)/len(rows),conditional_alpha=positions,
        accepted_draft_histogram=dict(Counter(r['accepted_drafts']for r in rows)),
        note='Worker cycles include warmup/terminal cycles unless the caller explicitly selects mature cycles; not a TPS measurement.')

def routes(frame,expected_layers):
    if set(map(int,frame['layers']))!=set(expected_layers):raise ValueError('incomplete target layer coverage')
    n=sum(count for _,count in frame['requests']);result=[]
    for layer,rows in frame['layers'].items():
        if len(rows)!=n or any(len(row)!=8 or len(set(row))!=8 or any(not 0<=v<384 for v in row)for row in rows):
            raise ValueError('bad or padded route rows')
        counts=Counter(v for row in rows for v in row)
        per_request=[];offset=0
        for rid,count in frame['requests']:
            group=Counter(v for row in rows[offset:offset+count]for v in row);offset+=count
            per_request.append(dict(request_id=rid,T=count,U=len(group),m_histogram=dict(Counter(group.values()))))
        result.append(dict(layer=int(layer),rows=n,U=len(counts),routes=8*n,m_histogram=dict(Counter(counts.values())),
                           maximum_logical_weight_saving=1-len(counts)/(8*n),per_request=per_request))
    return result

def budget(tau,step_ms):
    if tau<1 or step_ms<=0:raise ValueError('invalid cycle budget')
    return dict(per_request_tps=1000*tau/step_ms,maximum_step_ms_at_170=1000*tau/170,
                maximum_step_ms_at_100=10*tau,maximum_step_ms_at_80=12.5*tau)
