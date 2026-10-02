"""Interior SPMU samples for exact timed recipe/core intervals; no cycle percentages."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,re,statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--profile-analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
trace,=list((a.case/'trace').glob('*.json'));events=json.loads(trace.read_text())['traceEvents']
profile=json.loads(a.profile_analysis.read_text());assert profile['trace_sha256']==hashlib.sha256(trace.read_bytes()).hexdigest()
labels={};updates=defaultdict(list)
for e in events:
    if e['ph']=='C' and e.get('cat')=='SPMU':
        index=int(re.match(r'\[(\d+)\]',e['name'])[1]);labels[index]=e['name']
        updates[e['tid']].append((e['ts'],index,e['args'][e['name']]))
groups={};gaps=[]
for tid,records in updates.items():
    buckets=[]
    for t,index,value in sorted(records):
        if not buckets or t-buckets[-1]['start']>.2:buckets.append(dict(start=t,end=t,updates={}))
        buckets[-1]['end']=t;buckets[-1]['updates'][index]=value
    current={i:0 for i in labels}
    for b in buckets:current.update(b['updates']);b['values']=current.copy()
    gaps += [b['start']-prev['start'] for prev,b in zip(buckets,buckets[1:]) if 1<b['start']-prev['start']<10]
    groups[tid]=buckets
period=statistics.median(gaps);intervals=defaultdict(list)
selected={'gk_moe_gp_scale_tail_v1','downact_direct_down_broadcast','gk_moe_down_242_scale_tail_experiment','pf_combine_f32'}
for r in profile['ordered_replay_samples']:
    if r['phase']!='timed':continue
    for node in r['nodes']:
        if node['op']not in selected:continue
        for pid,tid,start,end in node['engine_intervals']:
            intervals[(r['variant'],node['op'])].append((tid,start+r['start_us'],end+r['start_us']))
nodes=[]
for (variant,op),spans in intervals.items():
    samples=[];covered=0;hit_intervals=0
    for tid,start,end in spans:
        count=0;bs=groups.get(tid,[])
        for previous,current in zip(bs,bs[1:]):
            if current['start']<start:continue
            if previous['start']>end:break
            gap=current['start']-previous['start']
            if previous['start']>=start+.1 and current['end']<=end-.1 and abs(gap-period)<=.2:
                samples.append(current['values']);covered+=gap;count+=1
        hit_intervals+=bool(count)
    values={}
    for index in labels:
        data=[s[index]for s in samples]
        values[index]=dict(mean=statistics.mean(data),median=statistics.median(data),min=min(data),max=max(data)) if data else None
    nodes.append(dict(variant=variant,op=op,core_intervals=len(spans),
        core_intervals_with_interior_samples=hit_intervals,interior_samples=len(samples),
        interior_coverage_fraction=covered/sum(end-start for _,start,end in spans),raw_values=values))
spmu,=list((a.case/'trace').glob('*spmu.res'));raw=spmu.read_text()
assert raw.count('Overflow flag : 0')==24
totals57=[int(n.replace(',',''))for n in re.findall(r'([\d,]+)\s*:\s*\[57\]',raw)]
assert len(totals57)==24
report=dict(trace_sha256=profile['trace_sha256'],decoded_labels=labels,
    raw_counter_events=sum(len(v)for v in updates.values()),requested_period_cycles=256,
    observed_group_period_us=period,store_tensor_full_counter57_totals=totals57,
    overflow_flags_zero=24,nodes=nodes,
    scope='Forward-filled raw decoded C values in wholly interior windows of actual timed core intervals only. Counters overlap. Requested256 is not a verified denominator; no clock/utilization/time percentages. Zero counter57 is not proof of no memory stalls. Short consumers may have no interior coverage.')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
