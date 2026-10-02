"""Build independently named scalar and vector post-TopK public graph nodes."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--torch',action='store_true');p.add_argument('--torch-only',action='store_true',help='rebuild only the public bridge; reuse independently pinned TPC library');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
identity=json.loads((root/'source-identity.json').read_text()) if not (root/'.git').exists() else None
commit=identity['git_commit'] if identity else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
sources=['csrc/tpc/moe_router/post_top8.c','csrc/tpc/moe_router/post_top8_vector.c','csrc/tpc/moe_router/precise_ratio.h','csrc/host/moe_router_post_glue.cpp','csrc/torch/moe_router_post.cpp','benchmarks/moe_router/simulate_post.cpp','tools/build_router_post.py']
for name in sources:
 data=(root/name).read_bytes()
 if identity:assert hashlib.sha256(data).hexdigest()==identity['files_sha256'][name]
 else:assert subprocess.check_output(['git','show',commit+':'+name],cwd=root)==data,'commit sources first'
 dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
report={'status':'BUILDING','source_commit':commit,'committed_sources_verified':True,'device_used':False,'commands':[]}
def run(cmd):
 report['commands'].append(cmd)
 with (out/'build.log').open('a') as log:log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
try:
 if not a.torch_only:
  for mode,name in [('scalar','post_top8.c'),('vector','post_top8_vector.c')]:
   for action,suffix in [('-c','o'),('-S','s')]:run(['/usr/bin/tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',action,str(root/'csrc/tpc/moe_router'/name),'-o',mode+'.'+suffix])
   run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',mode+'.o',mode+'_x86.o'])
   with (out/(mode+'.objdump')).open('w') as f:subprocess.run(['tpc-llvm-objdump','--triple=tpc','-d',str(out/(mode+'.o'))],stdout=f,check=True,timeout=30)
  run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',str(root/'csrc/host/moe_router_post_glue.cpp'),'scalar_x86.o','vector_x86.o','-o','libgaudi_router_post8_tpc.so'])
  run(['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(root/'benchmarks/moe_router/simulate_post.cpp'),'-L/usr/lib/habanatools','-Wl,-rpath,/usr/lib/habanatools','-ltpc_tests_core_ext','-o','simulator'])
 if a.torch or a.torch_only:
  os.environ['PT_HPU_LAZY_MODE']='1';os.environ.setdefault('MAX_JOBS','2')
  import habana_frameworks.torch as ht
  from torch.utils.cpp_extension import load
  h=Path(ht.__file__).parent
  load(name='gaudi_router_post8',sources=[str(root/'csrc/torch/moe_router_post.cpp')],extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
 report['status']='COMPILED_NOT_DEVICE_QUALIFIED'
except Exception as e:report.update(status='FAIL',error=repr(e));raise
finally:
 report['files_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
 (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'output':str(out)}))
