"""Complete route-tile graph timing, including all compiler split nodes."""
import argparse,json,statistics,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qkv_postprocess'))
from analyze_chain_profile import analyze
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
q=a.case/'post_graph.json';names=[]
for f in ([q]if q.is_file()else q.rglob('*.json')):
    for g in json.loads(f.read_text())['graphs']:
        if any(n['guid']=='gk_mxfp4_graph_decode_historical'for n in g['nodes']):names.append(g['name'])
assert len(names)==1
trace=list((a.case/'trace').glob('*.json'));assert len(trace)==1
x=analyze(trace[0],q,names,operation_aliases={'batch_gemm':'BatchGemm','gemm':'GEMM'})
rows=sorted(x['invocations'],key=lambda r:r['start_us']);gate=json.loads((a.case/'result.json').read_text());assert gate['status']=='PASS_TP_LOCAL_MOE_FP32_BOUND'
expected=1+len(gate['checks'])+sum(v.get('replays',1)for v in gate['timing'])
assert len(rows)==expected,(len(rows),expected)
timed=rows[1+len(gate['checks']):]
def union(xs):
    end=None;total=0
    for lo,hi in sorted(xs):
        if end is None or lo>=end:total+=hi-lo;end=hi
        elif hi>end:total+=hi-end;end=hi
    return total
records=[]
for row in timed:
    intervals=defaultdict(list);counts=defaultdict(int);sums=defaultdict(float)
    for node in row['nodes']:
        op=node['op'];counts[op]+=1;sums[op]+=node['envelope_us']
        intervals[op].extend((v[2],v[3])for v in node['engine_intervals'])
    records.append(dict(envelope_us=row['envelope_us'],physical_nodes=row['physical_nodes'],
        node_counts=dict(counts),op_union_us={k:union(v)for k,v in intervals.items()},op_envelope_sum_us=dict(sums),
        scope='Per-family unions may overlap; summed node envelopes are not latency decomposition'))
summary=dict(status='PASS_COMPLETE_NATIVE_PROFILE',timed_invocations=len(timed),qualification_invocations=1+len(gate['checks']),
    envelope_median_us=statistics.median(r['envelope_us']for r in records),
    op_union_median_us={k:statistics.median(r['op_union_us'][k]for r in records)for k in records[0]['op_union_us']},
    physical_node_counts=records[0]['node_counts'],records=records,trace_sha256=x['trace_sha256'])
a.output.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k!='records'},indent=2))
