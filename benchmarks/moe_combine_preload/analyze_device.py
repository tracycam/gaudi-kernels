"""Recheck all sealed full-chain outputs and keep every paired timing arm."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch

p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2)
manifest=json.loads((a.archive/'remote-sha256.json').read_text())
def checked(path):
 assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][str(path.relative_to(a.archive))]['sha256'],path
 return path
case=a.archive/'results'/a.case
exit_record=json.loads(checked(case/'exit.json').read_text());assert exit_record['runner_exit_code']==0 and exit_record['postflight']['returncode']==0
result=json.loads(checked(case/'result.json').read_text());assert result['status']=='PASS_FULL_CHAIN_BITS_AND_ABBA'
refs={};numeric_words=0;timing_words=0
for n in (1,2,8):
 for state in ('initial','hot_changed','cold_changed','zero_routes'):
  # Our own tensors, SHA verified before accepting the pickle payload.
  data=torch.load(checked(case/f'm{n}'/(state+'.pt')),weights_only=False)
  ref=data['variants']['deployed'][0];refs[n,state]=ref
  assert set(data['variants'])=={'deployed','control','unroll8'}
  for variant,samples in data['variants'].items():
   assert len(samples)==10
   for sample in samples:
    assert len(sample)==len(ref)==2
    for x,y in zip(ref,sample):
     assert y.dtype==torch.float32 and y.shape==(n,6144) and torch.isfinite(y).all()
     assert torch.equal(x.view(torch.uint8),y.view(torch.uint8))
     numeric_words+=y.numel()
  if state=='zero_routes':assert torch.equal(ref[0],torch.zeros_like(ref[0])) and torch.equal(ref[1],torch.full_like(ref[1],.03125))
assert len(result['timing'])==108
pairs=[]
for n in (1,2,8):
 for state in ('initial','hot_changed','cold_changed'):
  trials=[]
  for trial in range(3):
   arms=[r for r in result['timing']if (r['rows'],r['state'],r['trial'])==(n,state,trial)]
   assert [r['arm_index']for r in arms]==[0,1,2,3]
   assert [r['variant']for r in arms]==['deployed','unroll8','unroll8','deployed']
   for row in arms:
    got=torch.load(checked(case/row['output_file']),weights_only=False)
    assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8))for x,y in zip(refs[n,state],got))
    assert row['output_bits_equal'] and row['stable_logical_handles']
    assert [hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()for t in got]==row['output_sha256']
    assert .5<row['event_us']/row['wall_us']<1.2
    timing_words+=sum(v.numel()for v in got)
   old=statistics.mean(r['event_us']for r in arms if r['variant']=='deployed')
   new=statistics.mean(r['event_us']for r in arms if r['variant']=='unroll8')
   trials.append(dict(trial=trial,old_event_us=old,new_event_us=new,latency_delta_us=new-old,relative_delta=new/old-1))
  med={v:statistics.median(r['event_us']for r in result['timing']if (r['rows'],r['state'],r['variant'])==(n,state,v))for v in ('deployed','unroll8')}
  pairs.append(dict(rows=n,state=state,event_medians_us=med,relative_median_delta=med['unroll8']/med['deployed']-1,trials=trials))
post=case/'post_graph.json';graphs=[];controls=[];deployed=[]
for path in [post]if post.is_file()else post.rglob('*.json'):
 for graph in json.loads(checked(path).read_text())['graphs']:
  selected=[n for n in graph['nodes']if n['guid']=='gk_moe_combine_vector_routes_unroll8_experiment']
  if any(n['guid']=='gk_moe_combine_scalar_control_experiment'for n in graph['nodes']):controls.append(graph['name'])
  if selected:graphs.append(dict(name=graph['name'],candidate_nodes=len(selected),physical_nodes=sum(not n['is_logical']for n in graph['nodes'])))
assert len(graphs)==3 and len(controls)==3 and all(g['candidate_nodes']==1 for g in graphs)
report=dict(status='PASS_RECHECKED_FULL_CHAIN_BITS_AND_TIMING',numeric_words=numeric_words,timing_words=timing_words,
            comparisons=pairs,compiled_candidate_graphs=graphs,source_commit=manifest['source_commit'],
            disposition='Experimental combine-only comparison. No production promotion or model TPS claim.',
            scope='Original-width synthetic E24 top8 TP-local entire GP/gate/down/combine graph. Event spans include submission; not isolated combine cycles, physical HBM utilization or model TPS.')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ('comparisons','compiled_candidate_graphs')}))
