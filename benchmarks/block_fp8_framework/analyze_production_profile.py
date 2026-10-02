"""Actual current-precision QKV engine intervals, including TPC/MME overlap."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qkv_postprocess'))
from analyze_chain_profile import analyze
from audit import audit

def union(intervals):
    result=0.;end=-float('inf')
    for a,b in sorted(intervals):
        result+=max(0.,b-max(a,end));end=max(end,b)
    return result

p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
result=json.loads((a.case/'result.json').read_text())
assert result['status']=='PASS_NUMERICAL_PLACEMENT_UNAUDITED'
post=a.case/'post_graph.json';files=sorted(post.rglob('*.json'))if post.is_dir()else[post]
graphs=[g for f in files for g in json.loads(f.read_text())['graphs']
        if any(n['engine']=='MME'for n in g['nodes'])]
assert len(graphs)==len(result['records']) and len(graphs)>0
mapping={};engines={};placements={}
for g in graphs:
    ts={t['name']:t for t in g['tensors']};mmes=[n for n in g['nodes']if n['engine']=='MME']
    shapes={tuple(ts[n['input_tensors'][0]]['max_shape'])for n in mmes}
    # BF16 GEMM consumes [K,M]; FP8 block batchMME consumes [128,M,Gslice].
    assert len({s[1]for s in shapes})==1
    assert all((len(s)==2 and s[0]==6144)or(len(s)==3 and s[0]==128)for s in shapes)
    mapping[g['name']]=next(iter(shapes))[1];engines[g['name']]={int(n['id']):n['engine']for n in g['nodes']}
    placement=audit(g);assert placement['no_logical_weight_read_amplification']
    placements[g['name']]=placement
trace=list((a.case/'trace').glob('*.json'));assert len(trace)==1
raw=analyze(trace[0],post,list(mapping),operation_aliases={'gemm':'GEMM','batch_gemm':'BatchGemm'})
summaries=[]
for name,m in mapping.items():
    rows=sorted((r for r in raw['invocations']if r['recipe']==name),key=lambda r:r['start_us'])
    record=next(r for r in result['records']if r['case']['shape'][0]==m)
    repeats=record['case']['repeats'];assert repeats==8
    # Exact driver: capture, initial, changed-input, five warmups, 3x8 timed.
    assert len(rows)==8+3*repeats,(name,len(rows))
    rows=rows[8:];metrics=[]
    for row in rows:
        groups=defaultdict(list)
        for node in row['nodes']:
            kind=engines[name][node['node_id']]
            groups[kind]+=[(start,end)for _,_,start,end in node['engine_intervals']]
        tpc,mme=union(groups['TPC']),union(groups['MME']);busy=union(groups['TPC']+groups['MME'])
        metrics.append({'envelope_us':row['envelope_us'],'tpc_union_us':tpc,'mme_union_us':mme,
                        'overlap_us':tpc+mme-busy,'no_compute_us':row['envelope_us']-busy})
    summary={k:statistics.median(x[k]for x in metrics)for k in metrics[0]}
    summary.update(m=m,recipe=name,invocations=len(rows),placement=placements[name],
                   activation=record['case']['activation'],scale_math=record['case']['scale_math'])
    summary['whole_device_TFLOPS']=2*m*3392*6144/(summary['envelope_us']*1e6)
    summary['mme_union_TFLOPS']=2*m*3392*6144/(summary['mme_union_us']*1e6)
    summaries.append(summary)
raw['summary']=sorted(summaries,key=lambda r:r['m'])
raw['sequence_assignment']='8 qualification/warmup then24 timed invocations per shape, exact physical node coverage'
raw['measurement_scope']='Replay of actual rank0 QKV fixture; explicit activation/scale policy recorded per case, FP32 accumulation. No model TPS, clock-normalized utilization or physical HBM traffic claim.'
a.output.write_text(json.dumps(raw,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items()if k!='placement'}for r in raw['summary']],indent=2))
