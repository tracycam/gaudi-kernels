"""Same-data recipe comparisons and bounded in-recipe engine attribution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import numpy as np
from run import fixture,bf16_float

p=argparse.ArgumentParser()
p.add_argument('--build',type=Path,required=True)
p.add_argument('--grid',choices=['main','profile','tails','stream','ties'],default='main')
p.add_argument('--modes',nargs='+',default=['w8a8','w8a16'])
p.add_argument('--profile',action='store_true')
p.add_argument('--compare',type=Path)
a=p.parse_args();out=Path(os.environ['PROBE_OUT']);binary=a.build.resolve()/'fp8_linear_bench'
shapes={'main':[(m,1024,2048,'random') for m in [1,512,513]],
 'profile':[(512,1024,2048,'random')],
 'tails':[(2,255,513,'range'),(8,257,769,'cancellation'),(1,135,255,'zero'),(3,63,65,'random')],
 'ties':[(2,129,257,'ties'),(8,257,2049,'ties'),(2,129,257,'subnormal')],
 'stream':[(1,8192,8192,'random'),(64,4096,4096,'random')]}
records=[]
for m,n,k,kind in shapes[a.grid]:
 f=out/f'fixture-{m}-{n}-{k}-{kind}';refs=fixture(f,m,n,k,kind)
 for mode in a.modes:
  label=f'{mode}-{m}-{n}-{k}-{kind}';case=out/label;case.mkdir();graphs=case/'graphs';graphs.mkdir()
  env=dict(os.environ,HABANA_LOGS=str(case/'logs'),ENABLE_EXPERIMENTAL_FLAGS='true',
   DUMP_POST_GRAPHS=str(case/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(graphs),SRAM_SLICER_GRAPH_VISUALIZATION='1')
  if a.profile:
   trace=case/'trace';trace.mkdir();config=case/'profile.json'
   command=['hl-prof-config','--gaudi2','--config-filename',str(config),'-e','off','-o',str(trace),
    '-s','fp8-overhead','--phase','multi-enq','-g','7-9','-b','64','--invoc','json',
    '--merged','json,hltv','--trace-analyzer','on','--trace-analyzer-csv','on','--host','on','--add-pid','off']
   configured=subprocess.run(command,capture_output=True,text=True)
   (case/'config.log').write_text(configured.stdout+configured.stderr)
   if configured.returncode:raise RuntimeError('profiler configuration failed')
   env.update(HABANA_PROFILE='1',HABANA_PROF_CONFIG=str(config))
  command=[str(binary),mode,str(m),str(n),str(k),str(f)]
  with (case/'run.log').open('w') as log:
   result=subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
  rows=[json.loads(line) for line in (case/'run.log').read_text().splitlines() if line.startswith('{')]
  rec={'mode':mode,'M':m,'N':n,'K':k,'kind':kind,'returncode':result.returncode,'profiled':a.profile,
   'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'rows':rows,
   'controls':{key:value for key,value in env.items() if key.startswith('GK_')}}
  if (case/'output.bin').exists():
   raw=np.fromfile(case/'output.bin',np.uint16);actual=bf16_float(raw).astype(np.float64).reshape(m,n)
   rec['full_output_metrics']={name:{'relative_l2':float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-30)),
    'max_abs':float(np.max(np.abs(actual-ref)))} for name,ref in refs.items() if name!='sumabs_fp64'}
   if a.compare:
    old=np.fromfile(a.compare/label/'output.bin',np.uint16)
    rec['changed_bf16_outputs']=int(np.count_nonzero(raw!=old))
    rec['max_abs_change']=float(np.max(np.abs(bf16_float(raw)-bf16_float(old))))
  times=[r for r in rows if r.get('stage')=='timing']
  if times:
   us=statistics.median(r['event_us'] for r in times)
   rec.update(event_median_us=us,wall_median_us=statistics.median(r['wall_us'] for r in times),
    effective_tflops=2*m*n*k/us/1e6,weight_payload_gb_s=n*k/us/1000)
  (case/'result.json').write_text(json.dumps(rec,indent=2));records.append(rec)
  (out/'summary.json').write_text(json.dumps(records,indent=2))
  print(json.dumps({key:value for key,value in rec.items() if key not in ['rows','full_output_metrics']}),flush=True)
if any(r['returncode'] for r in records):raise SystemExit(1)
