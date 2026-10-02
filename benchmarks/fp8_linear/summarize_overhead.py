"""Combine immutable same-data comparisons and measured node-envelope unions."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from analyze_timeline import length
from run import bf16_float

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
records=[];failures=[];profiles=[]
for summary in sorted(a.root.glob('*/summary.json')):
 for r in json.loads(summary.read_text()):
  if r['returncode']:failures.append({'case':summary.parent.name,**r})
for candidate,baseline in [('rows-main-f','baseline-main-b'),('rows-tails-f','baseline-tails-f'),
 ('rows-ties-f','baseline-ties-f'),('rows-stream-f','baseline-stream-f')]:
 c=a.root/candidate;b=a.root/baseline
 if not (c/'summary.json').exists():continue
 for r in json.loads((c/'summary.json').read_text()):
  label=f"{r['mode']}-{r['M']}-{r['N']}-{r['K']}-{r['kind']}"
  old=json.loads((b/label/'result.json').read_text())
  fixture=f"fixture-{r['M']}-{r['N']}-{r['K']}-{r['kind']}"
  current_hashes=json.loads((c/fixture/'fixture.json').read_text())['sha256']
  old_hashes=json.loads((b/fixture/'fixture.json').read_text())['sha256']
  assert current_hashes==old_hashes,(candidate,label,'input mismatch')
  output=np.fromfile(c/label/'output.bin',np.uint16);previous=np.fromfile(b/label/'output.bin',np.uint16)
  assert output.size==previous.size==r['M']*r['N']
  gate=next(v for v in r['rows'] if v['stage']=='correctness')
  rec={'candidate':candidate,'baseline':baseline,'label':label,'same_fixture_sha256':True,
   'outputs':int(output.size),'gate_pass':bool(gate['pass']),'returncode':r['returncode'],
   'changed_bf16_outputs':int(np.count_nonzero(output!=previous)),
   'max_abs_change':float(np.max(np.abs(bf16_float(output)-bf16_float(previous)))),
   'baseline_us':old.get('event_median_us'),'candidate_us':r.get('event_median_us'),
   'wall_us':r.get('wall_median_us'),'effective_tflops':r.get('effective_tflops'),
   'weight_payload_gb_s':r.get('weight_payload_gb_s'),'metrics':r['full_output_metrics']}
  if rec['candidate_us']:rec['latency_reduction_fraction']=1-rec['candidate_us']/rec['baseline_us']
  records.append(rec)
for file in sorted(a.root.glob('profile-*/timeline.json')):
 for r in json.loads(file.read_text())['records']:
  stages={stage:statistics.median(length([(n['start_us'],n['end_us']) for n in it['nodes'] if stage in n['name']])
   for it in r['invocations']) for stage in ['quantize','decode','scale_bias','mme']}
  profiles.append({'case':file.parent.name,'route':r['case'],'stage_union_medians_us':stages,**r['median']})
result={'scope':'whole native recipes, complete same-data outputs; node-envelope unions are not ALU busy or physical bus counters',
 'records':records,'failed_candidates':failures,'profiles':profiles,
 'checked_candidate_outputs':sum(r['outputs'] for r in records),
 'all_candidate_gates_pass':all(r['gate_pass'] and r['returncode']==0 for r in records),
 'all_candidate_outputs_bit_identical':all(r['changed_bf16_outputs']==0 for r in records)}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['records','failed_candidates','profiles']}))
