"""Four bounded actual-ISA gates; no hardware or production installation."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build=a.build.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);records=[]
for t in (1,2):
 for mode in (0,1):
  d=out/f't{t}-mode{mode}';d.mkdir();cmd=[str(build/'simulator'),str(build/'libgaudi_down_scale_tail_tpc.so'),str(t),str(mode)];(d/'command.json').write_text(json.dumps(cmd)+'\n')
  with(d/'stdout.log').open('w')as f:r=subprocess.run(cmd,cwd=d,env={**os.environ,'TPC_RUNNER':'0'},stdout=f,stderr=subprocess.STDOUT,timeout=45)
  rows=[json.loads(line)for line in(d/'stdout.log').read_text().splitlines()if line.startswith('{')];records.append(dict(tokens=t,mode=mode,returncode=r.returncode,records=rows));(out/'partial.json').write_text(json.dumps(records,indent=2)+'\n');assert r.returncode==0,records[-1]
report=dict(status='PASS_ACTUAL_ISA_ONLY',records=records,device_tested=False,source_build=json.loads((build/'build.json').read_text())['source_commit'],library_sha256=hashlib.sha256((build/'libgaudi_down_scale_tail_tpc.so').read_bytes()).hexdigest(),files_sha256={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()for p in out.rglob('*')if p.is_file()})
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],words=sum(r['records'][-1]['words']for r in records))))
