"""Build isolated GUIDs from the live control ELF and rolled/unrolled new C."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from moe_activation_fold_core import embedded_elf
p=argparse.ArgumentParser();p.add_argument('--production-library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
pin='f3e6f643eff6482218c8c551b112df8f3e50740ca88a7e7200daae005015d1dc'
data=a.production_library.read_bytes()
if hashlib.sha256(data).hexdigest()!=pin:raise ValueError('Wrong deployed precision database')
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
sources=['csrc/tpc/moe_combine_preload/combine.c','csrc/host/moe_combine_preload_glue.cpp','tools/build_moe_combine_preload.py','tools/moe_activation_fold_core.py','benchmarks/moe_combine_preload/simulator.cpp','benchmarks/moe_combine_preload/run_sim.py','csrc/tpc/mxfp4_moe/legacy/combine_f32.c']
for name in sources:
 content=(root/name).read_bytes()
 if content!=subprocess.check_output(['git','show',commit+':'+name],cwd=root):raise ValueError('Commit source before build: '+name)
 target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
(out/'production.so').write_bytes(data);(out/'control.o').write_bytes(embedded_elf(data,'combine_f32'))
report=dict(status='BUILDING',source_commit=commit,device_tested=False,production_changed=False,commands=[])
def run(command):
 command=list(map(str,command));report['commands'].append(command)
 with(out/'build.log').open('a')as f:f.write(json.dumps(command)+'\n');f.flush();subprocess.run(command,cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
try:
 for stem,flags in [('candidate',[]),('unroll8',['-DGK_COMBINE_UNROLL8=1'])]:
  for option,extension in [('-c','o'),('-S','s')]:run(['tpc-clang','-O2','-mcpu=gaudi2',*flags,option,root/sources[0],'-o',stem+'.'+extension])
 for stem in ('control','candidate','unroll8'):
  run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',stem+'.o',stem+'_x86.o'])
  # GNU objcopy may rewrite its input even for a section dump. Never mutate
  # the full deployed control ELF after creating its binary object.
  run(['objcopy','--dump-section','.text='+stem+'.text',stem+'.o',stem+'-section-audit.o'])
  with(out/(stem+'.dis')).open('w')as f:subprocess.run(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',str(out/(stem+'.o'))],stdout=f,check=True,timeout=30)
 run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',root/sources[1],'control_x86.o','candidate_x86.o','unroll8_x86.o','-o','libgaudi_combine_preload_tpc.so'])
 for stem in ('control','candidate','unroll8'):
  if embedded_elf((out/'libgaudi_combine_preload_tpc.so').read_bytes(),stem)!=(out/(stem+'.o')).read_bytes():raise ValueError('Embedded ELF mismatch')
 run(['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',root/'benchmarks/moe_combine_preload/simulator.cpp','-L/usr/lib/habanatools','-Wl,-rpath,/usr/lib/habanatools','-ltpc_tests_core_ext','-ldl','-o','simulator'])
 report['status']='BUILT_OFFLINE_ONLY'
except Exception as exc:report.update(status='FAIL',error=repr(exc));raise
finally:
 report['sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()for p in out.rglob('*')if p.is_file()};(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
