"""Strict complete recipe/native engine pairing, including compiler slice nodes."""
import argparse,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qkv_postprocess'))
from analyze_chain_profile import analyze
p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
post=a.profile/'post_graph.json';files=list(post.rglob('*.json'))if post.is_dir()else[post]
gs=[g for p in files for g in json.loads(p.read_text())['graphs']];selected={}
for g in gs:
 ops=[n['guid']for n in g['nodes']]
 if 'gk_norm_block128_grid24_v1'in ops:selected['grid24']=g
 if 'gk_residual_rmsnorm_bf16_v1'in ops and 'batch_gemm'in ops:selected['separate']=g
assert set(selected)=={'grid24','separate'}
traces=list((a.profile/'trace').glob('*.json'));assert len(traces)==1
r=analyze(traces[0],post,[g['name']for g in selected.values()],{'batch_gemm':'BatchGemm'})
summary={}
for key,g in selected.items():
 rows=sorted([v for v in r['invocations']if v['recipe']==g['name']],key=lambda v:v['start_us'])
 # Capture/warm executions precede changed fixture and identical ABBA replay.
 # Preserve every invocation in raw report; warmed summary omits only first2.
 warm=rows[2:];ops=set(n['op']for v in warm for n in v['nodes'])
 def union(spans):
  total=0;end=-float('inf')
  for start,stop in sorted(spans):total+=max(0,stop-max(start,end));end=max(end,stop)
  return total
 summary[key]=dict(samples=len(warm),chain_median_us=statistics.median(v['envelope_us']for v in warm),
  physical_nodes=sorted(set(v['physical_nodes']for v in warm)),operations={})
 for op in ops:
  ns=[n for v in warm for n in v['nodes']if n['op']==op]
  summary[key]['operations'][op]=dict(node_envelope_median_us=statistics.median(n['envelope_us']for n in ns),
   node_count_per_recipe=sum(n['op']==op for n in warm[0]['nodes']),engines_per_node=sorted(set(n['engine_count']for n in ns)),
   all_slices_union_median_us=statistics.median(union([(b,e)for n in v['nodes']if n['op']==op for _,_,b,e in n['engine_intervals']])for v in warm))
r.update(summary=summary,operation_aliases={'batch_gemm':'BatchGemm'},
 attribution='Exact native recipe/node IDs; public batch_gemm is named BatchGemm in native trace. Physical slicing retained, not counted as one kernel.')
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(summary,indent=2))
