"""Bounded CPU ISA gates, saving failing exit codes and complete outputs."""
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--simulator',type=Path,required=True);p.add_argument('--old-elf',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
result={'device_used':False,'rows':[]}
for mode in (0,1):
 out=a.output/str(mode);out.mkdir();command=[str(a.simulator.resolve()),str(a.old_elf.resolve()),str(a.candidate.resolve()),str(mode)]
 try:
  r=subprocess.run(command,cwd=out,env=dict(os.environ,TPC_RUNNER='0'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=45)
  (out/'run.log').write_text(r.stdout);rows=[json.loads(line) for line in r.stdout.splitlines() if line.startswith('{')]
  result['rows'].append(dict(mode=mode,returncode=r.returncode,records=rows))
 except subprocess.TimeoutExpired as e:
  (out/'run.log').write_bytes(e.stdout or b'');result['rows'].append(dict(mode=mode,timeout=True))
result['all_pass']=all(r.get('returncode')==0 for r in result['rows']);result['qualification']='SIM only, not hardware issue or performance validation'
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(0 if result['all_pass'] else 1)
