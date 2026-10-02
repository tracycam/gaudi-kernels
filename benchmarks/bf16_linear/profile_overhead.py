"""Bounded module-selected native timeline capture, separate from speed timings."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[2];out=Path(os.environ['PROBE_OUT'])
build=root/os.environ.get('GK_BF16_BUILD','artifacts/builds/bf16-overhead/v1');binary=build/'bf16_linear_benchmark'
fixture=root/'results/bf16-overhead-full-v1/fixtures/random-128-128';records=[]
for backend,bias,depth in [('mme',0,1),('mme',1,1),('rowdot',1,1),('mme',0,16),('mme',1,16),('rowdot',1,16)]:
 label=f'{backend}-bias{bias}-depth{depth}';directory=out/label;directory.mkdir();trace=directory/'trace';trace.mkdir();config=directory/'profile.json'
 command=['hl-prof-config','--gaudi2','--config-filename',str(config),'-e','off','-o',str(trace),'-s','bf16','--phase','multi-enq','-g','9-11','-b','64','--invoc','json','--merged','json,hltv','--trace-analyzer','on','--trace-analyzer-csv','on','--host','on','--add-pid','off']
 configured=subprocess.run(command,capture_output=True,text=True);(directory/'config.log').write_text(configured.stdout+configured.stderr)
 rec={'case':label,'config_command':command,'config_returncode':configured.returncode}
 if configured.returncode==0:
  env=dict(os.environ,HABANA_PROFILE='1',HABANA_PROF_CONFIG=str(config),HABANA_LOGS=str(directory/'logs'),GK_BF16_CHAIN_DEPTH=str(depth))
  with (directory/'run.log').open('w') as stream:
   run=subprocess.run([str(binary),backend,'1','128','128','bf16',str(bias),'0',str(fixture),str(directory)],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
  rec['returncode']=run.returncode
 records.append(rec);print(json.dumps(rec),flush=True)
 (out/'result.json').write_text(json.dumps({'records':records,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'scope':'profiled invocation envelopes; use separate unprofiled recipe timing'},indent=2)+'\n')
assert all(r['config_returncode']==0 and r.get('returncode')==0 for r in records)
