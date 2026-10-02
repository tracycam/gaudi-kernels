"""Build fixed-row route-tile consumers; no device initialization unless --torch imports runtime.

The default local build produces TPC ELF/assembly, public glue; ISA simulation is a separate gate.
--torch builds the separate public bridge in an installed supported environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--torch',action='store_true');p.add_argument('--hoist-addresses',action='store_true');p.add_argument('--row-first',action='store_true');a=p.parse_args()
if a.row_first and not a.hoist_addresses:raise ValueError('row-first probe requires the qualified hoisted address baseline')
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
identity=json.loads((root/'source-identity.json').read_text())if not(root/'.git').exists()else None
commit=identity['git_commit']if identity else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
sources=['csrc/tpc/moe_route_tiles/'+n+'.c'for n in ('gather','gate','combine')]+['csrc/host/moe_route_tiles_glue.cpp','csrc/torch/moe_route_tiles.cpp','python/gaudi_kernels/mxfp4_route_tiles.py','tools/build_route_tiles.py']
sources += [str(p.relative_to(root))for p in sorted((root/'benchmarks/moe_route_tiles').glob('*'))if p.suffix in('.py','.cpp')]
for name in sources:
 data=(root/name).read_bytes()
 if identity:assert hashlib.sha256(data).hexdigest()==identity['files_sha256'][name]
 else:assert subprocess.check_output(['git','show',commit+':'+name],cwd=root)==data,'commit sources first'
 target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
report={'status':'BUILDING','source_commit':commit,'committed_sources_verified':True,'hoist_addresses':a.hoist_addresses,'row_first':a.row_first,'commands':[],'device_qualified':False}
def run(cmd):
 report['commands'].append(cmd)
 with(out/'build.log').open('a')as f:f.write(json.dumps(cmd)+'\n');f.flush();subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
try:
 for name in ('gather','gate','combine'):
  for action,suffix in [('-c','o'),('-S','s')]:run(['/usr/bin/tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',*(['-DGK_ROUTE_GATHER_HOIST=1']if a.hoist_addresses and name=='gather'else[]),*(['-DGK_ROUTE_GATHER_ROW_FIRST=1']if a.row_first and name=='gather'else[]),action,str(root/'csrc/tpc/moe_route_tiles'/f'{name}.c'),'-o',name+'.'+suffix])
  run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
  with(out/(name+'.dis')).open('w')as f:subprocess.run(['/usr/bin/tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',str(out/(name+'.o'))],stdout=f,check=True,timeout=30)
 run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',*(['-DGK_ROUTE_GATHER_ROW_FIRST=1']if a.row_first else[]),str(root/'csrc/host/moe_route_tiles_glue.cpp'),'gather_x86.o','gate_x86.o','combine_x86.o','-o','libgaudi_route_tiles_tpc.so'])
 if a.torch:
  if os.environ.get('PT_HPU_LAZY_MODE','1')!='1':raise ValueError('public Lazy CustomOp bridge requires PT_HPU_LAZY_MODE=1')
  os.environ.setdefault('PT_HPU_LAZY_MODE','1')
  os.environ.setdefault('MAX_JOBS','2')
  import habana_frameworks.torch as ht
  from torch.utils.cpp_extension import load
  h=Path(ht.__file__).parent
  load(name='gaudi_route_tiles',sources=[str(root/'csrc/torch/moe_route_tiles.cpp')],extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
 report['status']='COMPILED_NOT_DEVICE_QUALIFIED'
except Exception as e:report.update(status='FAIL',error=repr(e));raise
finally:
 report['files_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()for p in out.rglob('*')if p.is_file()}
 (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'source_commit':commit,'output':str(out)}))
