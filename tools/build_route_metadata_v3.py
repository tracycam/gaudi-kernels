"""Build fixed-shape route metadata; no device initialization unless --torch imports runtime.

The default local build produces TPC ELF/assembly, public glue and ISA simulator.
--torch builds the separate public bridge in an installed supported environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--torch',action='store_true');a=p.parse_args()
if a.torch and os.environ.get('PT_HPU_LAZY_MODE','1')!='1':p.error('--torch links the lazy public bridge; PT_HPU_LAZY_MODE must be1')
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
identity=json.loads((root/'source-identity.json').read_text())if not(root/'.git').exists()else None
commit=identity['git_commit']if identity else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
sources=['csrc/tpc/moe_route_metadata_v3/'+n+'.c'for n in ('count','inverse','row_map')]+['csrc/tpc/moe_route_metadata/prefix.c']+['csrc/host/moe_route_metadata_v3_glue.cpp','csrc/torch/moe_route_metadata_v3.cpp','python/gaudi_kernels/route_metadata_v3.py','python/gaudi_kernels/route_metadata.py','tools/build_route_metadata_v3.py']
sources += [str(p.relative_to(root))for p in sorted((root/'benchmarks/moe_route_metadata_v3').glob('*'))if p.suffix in('.py','.cpp')]
for name in sources:
 data=(root/name).read_bytes()
 if identity:assert hashlib.sha256(data).hexdigest()==identity['files_sha256'][name]
 else:assert subprocess.check_output(['git','show',commit+':'+name],cwd=root)==data,'commit sources first'
 target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
report={'status':'BUILDING','source_commit':commit,'committed_sources_verified':True,'commands':[],'device_qualified':False}
def run(cmd):
 report['commands'].append(cmd)
 with(out/'build.log').open('a')as f:f.write(json.dumps(cmd)+'\n');f.flush();subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
try:
 for name in ('count','prefix','inverse','row_map'):
  for action,suffix in [('-c','o'),('-S','s')]:run(['/usr/bin/tpc-clang','-O2','-mcpu=gaudi2',action,str(root/('csrc/tpc/moe_route_metadata'if name=='prefix'else'csrc/tpc/moe_route_metadata_v3')/f'{name}.c'),'-o',name+'.'+suffix])
  run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
  with(out/(name+'.dis')).open('w')as f:subprocess.run(['/usr/bin/tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',str(out/(name+'.o'))],stdout=f,check=True,timeout=30)
 run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',str(root/'csrc/host/moe_route_metadata_v3_glue.cpp'),'count_x86.o','prefix_x86.o','inverse_x86.o','row_map_x86.o','-o','libgaudi_route_metadata_v3_tpc.so'])
 run(['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(root/'benchmarks/moe_route_metadata_v3/simulator.cpp'),'-L/usr/lib/habanatools','-Wl,-rpath,/usr/lib/habanatools','-ltpc_tests_core_ext','-ldl','-o','simulator'])
 if a.torch:
  # The extension links the lazy public bridge; do not first load eager plugin.
  os.environ['PT_HPU_LAZY_MODE']='1'
  report['torch_build_lazy_mode']=1
  os.environ.setdefault('MAX_JOBS','2')
  import habana_frameworks.torch as ht
  from torch.utils.cpp_extension import load
  h=Path(ht.__file__).parent
  load(name='gaudi_route_metadata_v3',sources=[str(root/'csrc/torch/moe_route_metadata_v3.cpp')],extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
 report['status']='COMPILED_NOT_DEVICE_QUALIFIED'
except Exception as e:report.update(status='FAIL',error=repr(e));raise
finally:
 report['files_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()for p in out.rglob('*')if p.is_file()}
 (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'source_commit':commit,'output':str(out)}))
