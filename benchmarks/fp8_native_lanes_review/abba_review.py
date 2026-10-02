"""Recheck saved four-process ABBA tensors/timings without device access."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--canonical',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
c=a.canonical
old=c/'artifacts/builds/block-fp8-fp32-scale/module6-a/results'
new=c/'artifacts/builds/block-fp8-native-lanes/module6-a/results'
paths=[old/'lanes-abba-a0',new/'lanes-abba-b1',new/'lanes-abba-b2',old/'lanes-abba-a3']
driver=c/'artifacts/builds/block-fp8-native-lanes/abba-a/driver.json';plan=json.loads(driver.read_text())
assert [p['arm']for p in plan]==list('ABBA') and all(p['returncode']==0 for p in plan)
records=[json.loads((p/'result.json').read_text())for p in paths]
assert all(r['status']=='PASS_NUMERICAL_PLACEMENT_UNAUDITED'for r in records)
reported=json.loads((new/'native-lanes-abba-summary.json').read_text())
def bits(t):return t.contiguous().view(torch.uint8)
def fingerprint(value):
 if isinstance(value,torch.Tensor):return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(bits(value).numpy().tobytes()).hexdigest()}
 if isinstance(value,dict):return{k:fingerprint(v)for k,v in value.items()}
 if isinstance(value,(list,tuple)):return[fingerprint(v)for v in value]
 return value
summary=[]
for index,reference in enumerate(records[0]['records']):
 name=reference['case']['name'];reference_data=torch.load(paths[0]/name/'inputs-outputs.pt',map_location='cpu',weights_only=False)
 inputs=fingerprint({k:reference_data[k]for k in ('case','x','layers')});checks=[]
 for arm in range(1,4):
  got=torch.load(paths[arm]/name/'inputs-outputs.pt',map_location='cpu',weights_only=False)
  assert fingerprint({k:got[k]for k in ('case','x','layers')})==inputs,'different input/weight fixture'
  for field in ('actual','changed'):
   assert torch.equal(bits(got[field]),bits(reference_data[field])),(arm,name,field)
   checks.append(dict(case=paths[arm].name,tensor=field,all_bits_equal=True,words=got[field].numel()))
  del got
 events=[r['records'][index]['median_event_us']for r in records];walls=[r['records'][index]['median_wall_us']for r in records]
 assert events==reported[index]['ABBA_event_us']
 summary.append(dict(name=name,shape=reference['case']['shape'],ABBA_event_us=events,ABBA_wall_us=walls,
                     control_event_us=(events[0]+events[3])/2,candidate_event_us=(events[1]+events[2])/2,
                     control_wall_us=(walls[0]+walls[3])/2,candidate_wall_us=(walls[1]+walls[2])/2,
                     inputs=inputs,checks=checks))
 del reference_data
report=dict(status='PASS_RECHECKED_ABBA_OUTPUT_BITS',device_accessed=False,driver=plan,records=summary,
 source_files={str(p.relative_to(c)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [driver,new/'native-lanes-abba-summary.json',*[x/'result.json'for x in paths]]},
 scope='Four distinct processes A/B/B/A, fixed saved inputs/weights and initial/changed BF16 outputs. Mean of per-arm medians; HPUgraph full-recipe event and synchronized wall. Not same-process timing, isolated decoder time or whole-model gain.')
(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'output_words_checked':sum(x['words']for r in summary for x in r['checks'])}))
