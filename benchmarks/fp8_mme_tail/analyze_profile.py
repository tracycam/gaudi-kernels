"""Exact replay schedule + complete physical-node profile accounting."""
import argparse,json,statistics,sys
from pathlib import Path
from collections import defaultdict
from profile_events import analyze
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=json.loads((a.case/'result.json').read_text());assert d['status']=='PASS_ALL_OUTPUTS_PLACEMENT_ABBA';mapping={r['recipe']:r['padded_m']for r in d['captures']}
trace=list((a.case/'trace').glob('*_000-300.json'));assert len(trace)==1
raw=analyze(trace[0],a.case/'post_graph.json',list(mapping),operation_aliases={'gemm':'GEMM'})
def union(intervals):
    total=0.;end=-float('inf')
    for start,stop in sorted(intervals):total+=max(0.,stop-max(start,end));end=max(end,stop)
    return total
summaries=[]
for name,m in mapping.items():
    inv=sorted((x for x in raw['invocations']if x['recipe']==name),key=lambda r:r['start_us'])
    windows=[w for w in d['windows']if w['padded_m']==m]
    expected=3+sum(w['warmups']+w['replays']for w in windows)
    assert len(inv)==expected,(name,len(inv),expected)
    at=3;timed=[]
    for w in windows:
        at+=w['warmups'];timed+=inv[at:at+w['replays']];at+=w['replays']
    assert at==len(inv)
    metrics=[]
    for row in timed:
        groups=defaultdict(list)
        for node in row['nodes']:
            label='MME'if node['op']=='GEMM'else'DMA'if node['op']=='DmaMemcpy'else'decode'if node['op'].startswith('gk_block128_decode')else'finish'if node['op'].startswith('gk_block128_finish')else node['op']
            groups[label]+=[(s,e)for _,_,s,e in node['engine_intervals']]
        busy=union([x for xs in groups.values()for x in xs]);r={k+'_union_us':union(v)for k,v in groups.items()};r.update(envelope_us=row['envelope_us'],compute_union_us=union(groups['MME']+groups['decode']+groups['finish']),all_engine_union_us=busy,no_engine_us=row['envelope_us']-busy)
        metrics.append(r)
    summaries.append(dict(M=m,recipe=name,all_invocations=len(inv),timed_invocations=len(timed),median={k:statistics.median(x.get(k,0)for x in metrics)for k in set().union(*(x.keys()for x in metrics))},placement=next(c['placement']for c in d['captures']if c['recipe']==name)))
raw.update(status='PASS_COMPLETE_PROFILE_ASSIGNMENT',summary=sorted(summaries,key=lambda r:r['M']),assignment='Per recipe: capture+initial+changed, then exact recorded warmup/replay windows. All timed invocations retained; medians of overlapping groups must not be added.',performance_acceptance=False)
a.output.write_text(json.dumps(raw,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items()if k!='placement'}for r in raw['summary']],indent=2))
