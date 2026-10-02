"""Summarize bounded native device probes, keeping correctness and timing separate."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
for d in sorted((a.assets/'results').iterdir()):
 if not d.is_dir():continue
 meta=json.loads((d/'exit.json').read_text());fixture=json.loads((d/'fixture.json').read_text())
 events=[]
 for line in (d/'run.log').read_text().splitlines():
  if line.startswith('{'):
   try:events.append(json.loads(line))
   except json.JSONDecodeError:pass
 gate=next((x for x in events if x.get('stage')=='correctness'),None)
 compile=next((x for x in events if x.get('stage')=='compile_begin'),{})
 timing=[x for x in events if x.get('stage')=='timing']
 row={'case':d.name,'git_commit':meta['git_commit'],'mode':compile.get('mode'),'N':fixture['N'],'K':fixture['K'],'M':fixture['M'],'bias':fixture.get('bias',False),'splits':compile.get('splits'),'force_exact':compile.get('force_exact'),'profiled':d.name.startswith('profile-'),'exit_code':meta.get('runner_exit_code',meta.get('returncode')),'preflight_pass':meta.get('preflight',{}).get('returncode')==0,'postflight_pass':meta.get('postflight',{}).get('returncode')==0,'gate':gate,'sample_count':len(timing),'event_median_us':statistics.median(x['event_us'] for x in timing) if timing else None,'wall_median_us':statistics.median(x['wall_us'] for x in timing) if timing else None,'kernel_libraries_sha256':meta['kernel_libraries_sha256']}
 row['all_pass']=row['exit_code']==0 and gate is not None and gate['checked']>0 and gate['bad']==0 and len(timing)==5 and row['preflight_pass'] and row['postflight_pass']
 if row['mode']!='prepare':
  got=np.fromfile(d/'output.bin',np.uint32);want=np.fromfile(d/'expected.u32',np.uint32)
  assert len(got)==fixture['N']*fixture['M']==len(want)
  row['raw_output_bit_differences']=int(np.count_nonzero(got!=want));assert row['raw_output_bit_differences']==gate['bit_differences_from_exact']
  if row['mode']=='exact':assert np.array_equal(got,want)
  row['original_weight_scale_bytes']=fixture['packed_bytes']+fixture['scale_bytes']
  row['logical_weight_bytes_per_event_second_TBps']=row['original_weight_scale_bytes']/row['event_median_us']/1e6 if row['event_median_us'] else None
 if row['profiled']:
  profile=next((d/'trace').glob('*analyzed_nodes.csv'))
  nodes=[]
  for x in csv.DictReader(profile.open()):nodes.append({k:x[k] for k in ['Node Name','Duration (us)','Parallel Engines','Num activations','Start time of node','End time of node','Input placement','Output placement']})
  row['profile_nodes']=nodes
 rows.append(row)
comparisons=[]
for version in ['v1','v2','v3']:
 for shape in ['small','down','stream']:
  pred=a.assets/'results'/f'pred-{shape}-normal-{version}'/'output.bin';history=a.assets/'results'/f'history-{shape}-normal-v1'/'output.bin'
  if pred.is_file() and history.is_file():comparisons.append({'pred':str(pred.parent.name),'history':history.parent.name,'bit_identical':pred.read_bytes()==history.read_bytes()})
for v in ['v2','v3']:
 pred=a.assets/'results'/f'pred-bias-normal-{v}'/'output.bin';history=a.assets/'results/history-bias-normal-v2/output.bin'
 if pred.is_file() and history.is_file():comparisons.append({'pred':pred.parent.name,'history':history.parent.name,'bit_identical':pred.read_bytes()==history.read_bytes()})
result={'qualification':'native Gaudi2 module7 compact exact and aligned-M1 predispatch only; ordinary fast FP32 tolerance and exact bitwise gates remain distinct','all_pass':bool(rows) and all(x['all_pass'] for x in rows) and all(x['bit_identical'] for x in comparisons),'device_cases':len(rows),'physical_hbm_bytes_measured':False,'cases':rows,'same_input_history_comparisons':comparisons,'unqualified':['M>1 exact weight reuse','ragged three-node predispatch','physical HBM 1x traffic','framework replay of this exact/predispatch core','model throughput']}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['cases','same_input_history_comparisons']}))
