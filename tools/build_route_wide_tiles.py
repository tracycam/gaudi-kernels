"""Private C64/C128 glue around pinned, unchanged v3/row-first TPC ELF bytes.

The existing u4 core/bundle remains separately registered. No TPC compiler or
HPU acquisition is used here. --torch builds actual public wide-only interfaces.
"""
import argparse,ctypes,hashlib,json,os,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--metadata-library',type=Path,required=True);p.add_argument('--tiles-library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--torch',action='store_true');p.add_argument('--skip-simulators',action='store_true',help='target build need not install the offline ISA simulator library');p.add_argument('--abi-include',type=Path,default=Path('/usr/lib/habanatools/include'));a=p.parse_args()
if a.torch and os.getenv('PT_HPU_LAZY_MODE','1')!='1':p.error('Lazy bridge requires PT_HPU_LAZY_MODE=1')
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
identity=json.loads((root/'source-identity.json').read_text())if not(root/'.git').exists()else None
commit=identity['git_commit']if identity else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
names=['csrc/host/route_wide_metadata_glue.cpp','csrc/host/route_wide_tiles_glue.cpp','csrc/host/route_wide_bundle_glue.cpp','csrc/torch/route_wide_metadata.cpp','csrc/torch/route_wide_tiles.cpp','python/gaudi_kernels/route_wide_metadata.py','python/gaudi_kernels/mxfp4_route_wide_tiles.py','tools/build_route_wide_tiles.py','benchmarks/route_bundle/qualified_row_inputs.json','benchmarks/moe_route_tiles/isa_simulator.cpp']
names += [str(p.relative_to(root))for p in sorted((root/'benchmarks/moe_route_wide').glob('*'))if p.suffix in('.py','.cpp')]
for name in names:
 data=(root/name).read_bytes()
 if identity:assert sha(root/name)==identity['files_sha256'][name]
 else:assert data==subprocess.check_output(['git','show',commit+':'+name],cwd=root),'commit source first'
 dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
pins=json.loads((root/'benchmarks/route_bundle/qualified_row_inputs.json').read_text());apis=('GetKernelGuids','InstantiateTpcKernel','GetShapeInference','GetSupportedDataLayout','GetLibVersion')
report=dict(status='BUILDING',source_commit=commit,TPC_compiler_invoked=False,device_acquired=False,default_changed=False,commands=[],elfs={},input_libraries={})
def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
def run(command):
 command=list(map(str,command));report['commands'].append(command);save()
 with(out/'build.log').open('a')as f:f.write(json.dumps(command)+'\n');f.flush();subprocess.run(command,cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
try:
 objects=[]
 for role in('metadata','tiles'):
  lib=getattr(a,role+'_library').resolve();assert sha(lib)==pins[role]['library_sha256'],'unpinned source library'
  report['input_libraries'][role]=dict(path=str(lib),sha256=sha(lib));h=ctypes.CDLL(str(lib),mode=ctypes.RTLD_LOCAL)
  for name,expected in pins[role]['elfs'].items():
   begin=ctypes.addressof(ctypes.c_ubyte.in_dll(h,'_binary_'+name+'_o_start'));end=ctypes.addressof(ctypes.c_ubyte.in_dll(h,'_binary_'+name+'_o_end'));assert end-begin==expected['bytes'];blob=ctypes.string_at(begin,end-begin);assert hashlib.sha256(blob).hexdigest()==expected['elf_sha256']
   (out/(name+'.o')).write_bytes(blob);report['elfs'][name]=expected
   run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o']);objects.append(name+'_x86.o')
   with(out/(name+'.dis')).open('w')as f:subprocess.run(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',str(out/(name+'.o'))],stdout=f,check=True,timeout=30)
  command=['g++','-O2','-std=c++17','-fPIC','-I'+str(a.abi_include),*['-D'+api+'=gk_bundle_'+role+'_'+api for api in apis]]
  if role=='tiles':command+=['-DGK_ROUTE_GATHER_ROW_FIRST=1']
  run(command+['-c',root/'csrc/host'/('route_wide_'+role+'_glue.cpp'),'-o',role+'_glue.o']);objects.append(role+'_glue.o')
 run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+str(a.abi_include),root/'csrc/host/route_wide_bundle_glue.cpp',*objects,'-o','libgaudi_route_wide_tpc.so'])
 # Check the final database still contains every original full ELF verbatim.
 h=ctypes.CDLL(str(out/'libgaudi_route_wide_tpc.so'),mode=ctypes.RTLD_LOCAL)
 for name,expected in report['elfs'].items():
  begin=ctypes.addressof(ctypes.c_ubyte.in_dll(h,'_binary_'+name+'_o_start'));end=ctypes.addressof(ctypes.c_ubyte.in_dll(h,'_binary_'+name+'_o_end'));assert hashlib.sha256(ctypes.string_at(begin,end-begin)).hexdigest()==expected['elf_sha256']
 if not a.skip_simulators:
  for name,source in [('metadata_simulator','benchmarks/moe_route_wide/metadata_simulator.cpp'),('consumer_simulator','benchmarks/moe_route_tiles/isa_simulator.cpp')]:
   run(['g++','-O2','-std=c++17','-I'+str(a.abi_include),root/source,'-L/usr/lib/habanatools','-Wl,-rpath,/usr/lib/habanatools','-ltpc_tests_core_ext','-ldl','-o',name])
 if a.torch:
  os.environ['PT_HPU_LAZY_MODE']='1';os.environ.setdefault('MAX_JOBS','2')
  import habana_frameworks.torch as ht
  from torch.utils.cpp_extension import load
  h=Path(ht.__file__).parent
  load(name='gaudi_route_wide',sources=[str(root/'csrc/torch'/('route_wide_'+role+'.cpp'))for role in('metadata','tiles')],extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
 report['status']='BUILT_UNCHANGED_ELFS_NOT_DEVICE_QUALIFIED'
except Exception as e:report.update(status='FAIL',error=repr(e));raise
finally:
 report['files_sha256']={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name!='build.json'};save()
print(json.dumps({k:v for k,v in report.items()if k not in('commands','files_sha256','elfs','input_libraries')}))
