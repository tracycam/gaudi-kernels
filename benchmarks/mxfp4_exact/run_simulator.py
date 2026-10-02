"""Bounded local ISA checks; never selects an HPU runner."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);build=a.build.resolve();source=Path(__file__).with_name('simulator.cpp')
command=['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(source),'-L'+str(build),'-lmxfp4_exact_tpc','-L/usr/lib/habanatools','-ltpc_tests_core_ext','-Wl,-rpath,'+str(build)+':/usr/lib/habanatools','-o',str(out/'simulator')]
meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'compile_command':command,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256((build/'libmxfp4_exact_tpc.so').read_bytes()).hexdigest(),'TPC_RUNNER':'0','timeout_per_process_s':45,'device_validated':False,'cases':[]}
with (out/'compile.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
env=dict(os.environ,TPC_RUNNER='0');env['LD_LIBRARY_PATH']=str(build)+':/usr/lib/habanatools'
for case in ['exact-subnormal','exact-significant','exact-cancellation','exact-negative-half','history-unsafe','history-safe']:
 try:
  with (out/(case+'.log')).open('w') as log:result=subprocess.run([str(out/'simulator'),case],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=45)
  meta['cases'].append({'case':case,'returncode':result.returncode})
 except subprocess.TimeoutExpired:meta['cases'].append({'case':case,'timeout':True})
meta['all_pass']=all(case.get('returncode')==0 for case in meta['cases']);meta['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
(out/'result.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k!='artifacts'}));raise SystemExit(0 if meta['all_pass'] else 1)
