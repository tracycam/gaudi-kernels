"""Run real compiled candidate/control ISA against declared CPU FP32 orders."""
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--simulator',type=Path,required=True);a=p.parse_args();records=[]
fixture=json.loads((a.fixtures/'summary.json').read_text())
for case in fixture['records']:
 d=a.fixtures/case['name'];g,m,n=case['partial_shape']
 for chain in [1,4,8,0]:
  log=d/f'isa-chain{chain}.log';output=d/f'isa-chain{chain}-bf16.bin'
  with log.open('w') as f:
   result=subprocess.run([str(a.simulator.resolve()),str(chain),str(d.resolve()),str(m),str(n),str(g),str(output.resolve())],env={**os.environ,'TPC_RUNNER':'0'},stdout=f,stderr=subprocess.STDOUT,timeout=45)
  report=next((json.loads(s) for s in reversed(log.read_text().splitlines()) if s.startswith('{')),None)
  records.append(dict(case=case['name'],chains=chain,returncode=result.returncode,result=report))
  if result.returncode:raise RuntimeError('ISA failure retained at '+str(log))
 print(case['name'],flush=True)
report=dict(isa_matches_declared_cpu_order=all(r['returncode']==0 and r['result']['bit_mismatches_vs_declared_cpu_order']==0 for r in records),checked=sum(r['result']['checked'] for r in records),all_accuracy_gates_pass=fixture['all_accuracy_gates_pass'],production_qualified=False,device_tested=False,records=records)
(a.fixtures/'isa-summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
