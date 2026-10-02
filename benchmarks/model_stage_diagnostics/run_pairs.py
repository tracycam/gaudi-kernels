"""Run only CPU comparisons of the five stage pairs present in production-isa-70-a."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
pairs=[('native-vs-a8','cpu-native-a8-bf16-fp32','decode-a8-bf16-fp32'),
 ('native-vs-a8-swa','cpu-native-a8-bf16-fp32-swa-fp32-fast','decode-a8-bf16-fp32-gp-vec-down-vec-swa-fp32-fast'),
 ('a8-vs-gp','decode-a8-bf16-fp32','decode-a8-bf16-fp32-gp-vec'),
 ('gp-vs-down','decode-a8-bf16-fp32-gp-vec','decode-a8-bf16-fp32-gp-vec-down-vec'),
 ('down-vs-swa','decode-a8-bf16-fp32-gp-vec-down-vec','decode-a8-bf16-fp32-gp-vec-down-vec-swa-fp32-fast')]
for name,reference,candidate in pairs:
 with (a.out/(name+'.log')).open('w') as log:
  subprocess.run([sys.executable,str(Path(__file__).with_name('analyze.py')),str(a.case),'--reference',reference,'--candidate',candidate,'--out',str(a.out/(name+'.json'))],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
 print(name,flush=True)
