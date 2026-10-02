"""Experimental dual-GUID N512 down; original ELF control, text-only 242 candidate."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from moe_activation_fold_core import embedded_elf
p=argparse.ArgumentParser();p.add_argument('--canonical',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--torch',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
names=['tools/build_moe_down_scale_tail.py','tools/moe_activation_fold_core.py','benchmarks/moe_production_issue/audit_down_scale_tail.py','benchmarks/moe_production_issue/scale_tail.py','csrc/host/moe_down_scale_tail_glue.cpp','csrc/torch/moe_down_scale_tail.cpp']
names += [str(p.relative_to(root))for p in (root/'benchmarks/moe_down_scale_tail').glob('*')if p.suffix in('.py','.cpp')]
for name in names:
 assert(root/name).read_bytes()==subprocess.check_output(['git','show',commit+':'+name],cwd=root),'commit source before build: '+name
 target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/name).read_bytes())
report=dict(status='BUILDING',source_commit=commit,source_verified=True,device_acquired=False,production_changed=False,commands=[])
def run(cmd):
 report['commands'].append(list(map(str,cmd)))
 with(out/'build.log').open('a')as f:f.write(json.dumps(list(map(str,cmd)))+'\n');f.flush();subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
try:
 run([sys.executable,root/'benchmarks/moe_production_issue/audit_down_scale_tail.py','--canonical',a.canonical.resolve(),'--output',out/'schedule'])
 (out/'control.o').write_bytes((out/'schedule/production.o').read_bytes());(out/'candidate.o').write_bytes((out/'control.o').read_bytes());(out/'candidate.s').write_bytes((out/'schedule/proposed-uncompiled.s').read_bytes())
 run(['tpc-clang','-mcpu=gaudi2','-c','candidate.s','-o','candidate-assembled.o']);run(['objcopy','--dump-section','.text=candidate.text','candidate-assembled.o']);run(['objcopy','--update-section','.text=candidate.text','candidate.o'])
 for stem in ('control','candidate'):
  run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',stem+'.o',stem+'_x86.o'])
  with(out/(stem+'.dis')).open('w')as f:subprocess.run(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',str(out/(stem+'.o'))],stdout=f,check=True,timeout=30)
 run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',root/'csrc/host/moe_down_scale_tail_glue.cpp','control_x86.o','candidate_x86.o','-o','libgaudi_down_scale_tail_tpc.so'])
 for stem in ('control','candidate'):assert embedded_elf((out/'libgaudi_down_scale_tail_tpc.so').read_bytes(),stem)==(out/(stem+'.o')).read_bytes()
 run(['g++','-O2','-std=c++17','-ffp-contract=off','-I/usr/lib/habanatools/include',root/'benchmarks/moe_down_scale_tail/simulator.cpp','-L/usr/lib/habanatools','-Wl,-rpath,/usr/lib/habanatools','-ltpc_tests_core_ext','-ldl','-o','simulator'])
 if a.torch:
  if os.getenv('PT_HPU_LAZY_MODE','1')!='1':raise ValueError('lazy bridge requires PT_HPU_LAZY_MODE=1')
  os.environ['PT_HPU_LAZY_MODE']='1';os.environ.setdefault('MAX_JOBS','2')
  import habana_frameworks.torch as ht
  from torch.utils.cpp_extension import load
  h=Path(ht.__file__).parent
  load(name='gaudi_down_scale_tail',sources=[str(root/'csrc/torch/moe_down_scale_tail.cpp')],extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
 report['status']='BUILT_NOT_DEVICE_QUALIFIED'
except Exception as exc:report.update(status='FAIL',error=repr(exc));raise
finally:
 report['files_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()for p in out.rglob('*')if p.is_file()};(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
