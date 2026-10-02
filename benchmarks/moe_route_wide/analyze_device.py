"""Independent CPU reread of sealed wide-route outputs and complete ABBA arms."""
import argparse, hashlib, json, statistics
from pathlib import Path
import torch
from plan import STATES, VARIANTS, complete

p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2)
manifest_path=a.archive/'remote-sha256.json';manifest=json.loads(manifest_path.read_text())
assert manifest['status']=='REMOTE_LOCAL_SHA_VERIFIED'
hashes={};words=0
def read(path):
 name=str(path.relative_to(a.archive));record=manifest['files'][name]
 h=hashlib.sha256()
 with path.open('rb')as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 digest=h.hexdigest()
 assert digest==record['sha256'] and path.stat().st_size==record['bytes'],name
 hashes[name]=digest
 return torch.load(path,map_location='cpu',weights_only=False)
def equal(x,y):
 global words
 assert x.shape==y.shape and x.dtype==y.dtype
 words+=x.numel()
 assert torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
cases=[]
for case in sorted((a.archive/'results').iterdir()):
 if not case.is_dir():continue
 exit_record=json.loads((case/'exit.json').read_text());assert exit_record['returncode']==0 and exit_record['postflight']['returncode']==0
 result=json.loads((case/'result.json').read_text());assert result['status']=='PASS_CROSS_C_BITS_FULL_CHAIN_DIAGNOSTIC'
 assert result['numeric_complete'] and result['timing_complete'] and not result['candidate_accepted']
 tokens=result['tokens'];references={}
 states=result.get('states',STATES)
 if result.get('routing_pattern')=='uniform_all':
  fixture=read(case/'inputs.pt')
  expected=torch.arange(tokens*8,dtype=torch.int32).reshape(tokens,8)%384
  equal(fixture['original']['ids'],expected);equal(fixture['restored']['ids'],expected)
  equal(fixture['hot_routes']['ids'],torch.arange(8,dtype=torch.int32).expand(tokens,8))
  for state in states:
   assert torch.bincount(fixture[state]['ids'].flatten().long(),minlength=384).tolist()==result['expert_counts'][state]
  assert sum(v>0 for v in result['expert_counts']['original'])==384
 for state in states:
  values={}
  for variant in VARIANTS:
   samples=read(case/f'{state}-{variant}.pt');assert len(samples)==3
   for sample in samples:
    y,c=sample['y'],sample['consumer'];assert y.shape==(tokens,6144) and y.dtype==torch.float32
    assert torch.isfinite(y).all() and torch.isfinite(c).all();equal(c,y+.03125)
    equal(y,samples[0]['y']);equal(c,samples[0]['consumer'])
    if state=='zero_routes':equal(y,torch.zeros_like(y))
   values[variant]=samples[0];references[state,variant]=samples[0]
  for variant in ('c64','c128'):
   for key in ('y','consumer'):equal(values[variant][key],values['c32'][key])
 for variant in VARIANTS:
  for key in ('y','consumer'):equal(references['restored',variant][key],references['original',variant][key])
 plan=result.get('timing_plan',dict(anchor='broadcast',trials=3))
 assert complete(result['timing'],plan['trials'],plan['anchor'])
 for arm in result['timing']:
  value=read(case/arm['output_file']);ref=references['original',arm['variant']]
  for key in ('y','consumer'):equal(value[key],ref[key])
 pairs=[]
 for pair in sorted(set(r['pair']for r in result['timing'])):
  rows=[r for r in result['timing']if r['pair']==pair];trials=[]
  for t in range(plan['trials']):
   arms=[r for r in rows if r['trial']==t];assert [r['arm']for r in arms]==[0,1,2,3]
   base=statistics.mean(r['event_us']for r in arms if r['variant']==plan['anchor'])
   candidate=statistics.mean(r['event_us']for r in arms if r['variant']==pair)
   trials.append(dict(trial=t,baseline_us=base,candidate_us=candidate,ratio=candidate/base,saved_us=base-candidate))
  timing={v:dict(event_us=statistics.median(r['event_us']for r in rows if r['variant']==v),wall_us=statistics.median(r['wall_us']for r in rows if r['variant']==v))for v in (plan['anchor'],pair)}
  useful_flops=tokens*8*2*(6144*512+256*6144)
  pairs.append(dict(candidate=pair,anchor=plan['anchor'],timing=timing,trials=trials,median_ratio=statistics.median(r['ratio']for r in trials),useful_TFLOPS=useful_flops/timing[pair]['event_us']/1e6,useful_FLOP_scope='Only active expert linear MACs; excludes padding, activation and metadata. Not measured physical utilization.'))
 cases.append(dict(case=case.name,tokens=tokens,fixture_transform=result['fixture_transform'],pairs=pairs,checks=len(result['checks']),timing_arms=len(result['timing'])))
report=dict(status='ROOT_FULL_TENSORS_AND_TIMING_RECHECKED',archive=str(a.archive),manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),compared_words=words,tensor_sha256=hashes,cases=cases,production_accepted=False,model_quality_qualified=False,model_TPS_measured=False)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],compared_words=words,cases=[dict(case=c['case'],ratios=[(r['candidate'],r['median_ratio'])for r in c['pairs']])for c in cases])))
