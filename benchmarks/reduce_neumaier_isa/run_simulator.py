"""Bounded local TPC simulator comparison; never acquires an HPU."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--build', type=Path, required=True); p.add_argument('--fixtures', type=Path, required=True);p.add_argument('--modes', type=int,nargs='+',default=[0,1,2]); p.add_argument('--out', type=Path, required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
records=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary=dict(device_tested=False,simulator_sha256=sha(a.build/'simulator'),library_sha256=sha(a.build/'libreduce_neumaier_isa.so'),fixture_root=str(a.fixtures.resolve()),records=records)
try:
 for case in json.loads((a.fixtures/'summary.json').read_text())['records']:
  name=case['name'];g,m,n=case['partial_shape'];folder=a.fixtures/name
  for mode in a.modes:
   output=a.out/f'{name}-mode{mode}.bin';log=a.out/f'{name}-mode{mode}.log'
   command=[str((a.build/'simulator').resolve()),str(mode),str(folder.resolve()),str(m),str(n),str(g),str(output.resolve())]
   started=time.monotonic()
   try:
    r=subprocess.run(command,env=dict(os.environ,TPC_RUNNER='0'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=45)
    log.write_text(r.stdout);stats=next((json.loads(line) for line in r.stdout.splitlines() if line.startswith('{"checked"')), {})
    row=dict(case=name,mode=mode,exit_code=r.returncode,elapsed_seconds=time.monotonic()-started,command=command,**stats)
   except subprocess.TimeoutExpired as e:
    log.write_bytes(e.stdout or b'');row=dict(case=name,mode=mode,exit_code=124,elapsed_seconds=time.monotonic()-started,command=command)
   row['log_sha256']=sha(log)
   if output.exists():row['output_sha256']=sha(output)
   records.append(row);print(json.dumps(row),flush=True)
   summary['all_bits_pass']=all(r['exit_code']==0 and r.get('bit_mismatches_vs_declared_cpu_order')==0 for r in records)
   (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
finally:
 summary['complete']=len(records)==len(json.loads((a.fixtures/'summary.json').read_text())['records'])*len(a.modes)
 summary['all_bits_pass']=summary['complete'] and all(r['exit_code']==0 and r.get('bit_mismatches_vs_declared_cpu_order')==0 for r in records)
 summary['checked_total']=sum(r.get('checked',0) for r in records)
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
raise SystemExit(0 if summary['all_bits_pass'] else 1)
