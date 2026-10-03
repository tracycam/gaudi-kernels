"""Build device expert-M routing and consumers; no device acquisition."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--torch',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
sources={'prefix':'moe_partition/prefix.c','inverse':'moe_partition/inverse.c',
 'row_map':'moe_route_metadata_v3/row_map.c','gather':'moe_route_tiles/gather.c',
 'gate':'moe_route_tiles/gate.c','combine':'moe_partition/combine.c'}
identity=root/'source-identity.json'
commit=json.loads(identity.read_text())['git_commit'] if identity.exists() and not (root/'.git').exists() else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
names=['csrc/tpc/'+v for v in sources.values()]+['csrc/host/moe_partition_glue.cpp','csrc/torch/moe_partition.cpp','tools/build_moe_partition.py']
for name in names:
 data=(root/name).read_bytes()
 if identity.exists() and not (root/'.git').exists():assert hashlib.sha256(data).hexdigest()==json.loads(identity.read_text())['files_sha256'][name]
 else:assert data==subprocess.check_output(['git','show',commit+':'+name],cwd=root),'commit sources first'
report=dict(source_commit=commit,commands=[],state='building',device_acquired=False)
def run(cmd):
 report['commands'].append(list(map(str,cmd)))
 with (out/'build.log').open('a') as log:
  log.write(json.dumps(report['commands'][-1])+'\n');log.flush()
  subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
try:
 for name,source in sources.items():
  flags=['-DGK_ROUTE_GATHER_ROW_FIRST=1'] if name=='gather' else []
  run(['tpc-clang','-O2','-mcpu=gaudi2',*flags,'-c',str(root/'csrc/tpc'/source),'-o','pc_'+name+'.o'])
  run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','pc_'+name+'.o','pc_'+name+'_x86.o'])
  with (out/(name+'.dis')).open('w') as f:subprocess.run(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',str(out/('pc_'+name+'.o'))],stdout=f,check=True,timeout=30)
 run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/include/habanalabs',str(root/'csrc/host/moe_partition_glue.cpp'),*['pc_'+n+'_x86.o' for n in sources],'-o','libgaudi_expert_partition_tpc.so'])
 if a.torch:
  os.environ['PT_HPU_LAZY_MODE']='1';os.environ.setdefault('MAX_JOBS','2')
  import habana_frameworks.torch as ht
  from torch.utils.cpp_extension import load
  h=Path(ht.__file__).parent
  load(name='gaudi_expert_partition',sources=[str(root/'csrc/torch/moe_partition.cpp')],extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
 report['state']='built_not_device_verified'
except BaseException as error:
 report.update(state='failed',error=repr(error));raise
finally:
 report['files_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='build.json'}
 (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'state':report['state'],'output':str(out)}))
