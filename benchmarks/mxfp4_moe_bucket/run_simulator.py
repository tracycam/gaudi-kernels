"""Bounded offline ISA gate. Does not load Synapse or acquire an HPU."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2];out=a.build.resolve()
command=['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(root/'benchmarks/mxfp4_moe_bucket/simulator.cpp'),'-L'+str(out),'-lmxfp4_moe_bucket_tpc','-L/usr/lib/habanatools','-ltpc_tests_core_ext','-Wl,-rpath,'+str(out)+':/usr/lib/habanatools','-o',str(out/'simulator')]
subprocess.run(command,check=True);cases=[]
for case in ['ragged','boundaries','violation']:
 cmd=[str(out/'simulator'),case]
 with (out/('sim-'+case+'.log')).open('w') as log:
  try:proc=subprocess.run(cmd,env=dict(os.environ,TPC_RUNNER='0'),stdout=log,stderr=subprocess.STDOUT,timeout=45);code=proc.returncode
  except subprocess.TimeoutExpired:code=124
 rows=[json.loads(line) for line in (out/('sim-'+case+'.log')).read_text().splitlines() if line.startswith('{')]
 cases.append({'case':case,'command':cmd,'returncode':code,'result':rows[-1] if rows else None})
result={'compile_command':command,'kernel_sha256':hashlib.sha256((out/'libmxfp4_moe_bucket_tpc.so').read_bytes()).hexdigest(),'all_pass':all(c['returncode']==0 and c['result'] and c['result']['bad']==0 for c in cases),'cases':cases,'MME_emulated_on_CPU':True,'native_device_recipe_validated':False}
(out/'simulator-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'all_pass':result['all_pass'],'checked':sum(c['result']['checked'] for c in cases if c['result'])}));raise SystemExit(0 if result['all_pass'] else 3)
