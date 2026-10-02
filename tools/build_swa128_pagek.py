"""Build the isolated score-only page-K prototype and CPU ISA harness."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
sources=['csrc/tpc/swa128_pagek/qk.c','benchmarks/swa128_pagek/simulator.cpp','benchmarks/swa128_pagek/run_simulator.py','benchmarks/swa128_pagek/isa_budget.py','tools/build_swa128_pagek.py'];hashes={}
for name in sources:
 data=(root/name).read_bytes();assert data==subprocess.check_output(['git','show',commit+':'+name],cwd=root),'commit source first'
 hashes[name]=hashlib.sha256(data).hexdigest();target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
commands=[['/usr/bin/tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',mode,str(root/sources[0]),'-o',str(out/target)]for mode,target in [('-c','qk.o'),('-S','qk.s')]]
commands.append(['g++','-std=c++17','-O2',str(root/sources[1]),'-I/usr/lib/habanatools/include','-L/usr/lib/habanatools','-Wl,-rpath,/usr/lib/habanatools','-ltpc_tests_core_ext','-o',str(out/'simulator')])
record={'commit':commit,'sources_sha256':hashes,'commands':commands,'compiler':subprocess.check_output(['/usr/bin/tpc-clang','--version'],text=True)}
with(out/'build.log').open('w')as log:
 for command in commands:subprocess.run(command,check=True,stdout=log,stderr=log)
record['outputs_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in out.iterdir()if p.is_file()}
(out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
