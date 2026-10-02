import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cases=[]
for d in sorted((a.assets/'results').iterdir()):
 if not d.is_dir():continue
 if not (d/'exit.json').exists():cases.append({'case':d.name,'state':'incomplete_before_payload','all_pass':False});continue
 meta=json.loads((d/'exit.json').read_text());rows=[]
 for line in (d/'run.log').read_text().splitlines():
  if line.startswith('{'):
   try:rows.append(json.loads(line.replace(':inf',':Infinity').replace(':nan',':NaN')))
   except json.JSONDecodeError:pass
 gate=next((x for x in rows if x.get('stage')=='gate'),None);timing=[x for x in rows if x.get('stage')=='timing']
 if gate:
  for k,v in gate.items():
   if isinstance(v,float) and not math.isfinite(v):gate[k]=str(v)
 cases.append({'case':d.name,'source_commit':meta['git_commit'],'kernel_library_sha256':meta['kernel_libraries_sha256'],'runner_exit_code':meta.get('runner_exit_code',meta.get('returncode')),'gate':gate,'all_pass':meta.get('returncode')==0 and bool(gate) and gate['bad']==0,'event_median_us':statistics.median(x['event_us'] for x in timing) if timing else None,'wall_median_us':statistics.median(x['wall_us'] for x in timing) if timing else None})
base=a.assets/'results/t1-both-old-v4/output.bin';comparisons=[]
if base.exists():
 golden=np.fromfile(base,np.uint32)
 for name in ['t1-direct-v1','t1-fused256-v1','t1-fused256c-v2']:
  p=a.assets/'results'/name/'output.bin'
  if p.exists():comparisons.append({'case':name,'baseline':'t1-both-old-v4','bit_differences':int(np.count_nonzero(np.fromfile(p,np.uint32)!=golden))})
r={'all_pass':all(x['all_pass'] for x in cases),'production_promotion':False,'production_selection':'retain literal direct_down plus precision_fix FP32 combine','native_cases':sum('gate' in x for x in cases),'cases':cases,'full_output_comparisons':comparisons,'physical_hbm_measured':False,'reason':'N256 fusion gates pass but full recipe regresses; N512 private-VLM fusion fails hardware gates despite passing local instruction simulation'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['cases','full_output_comparisons']}))
