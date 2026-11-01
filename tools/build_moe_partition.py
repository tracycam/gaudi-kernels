"""Build device expert-M routing and consumers; no device acquisition."""
import argparse,hashlib,json,os,subprocess,shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--torch',action='store_true');p.add_argument('--jobs',type=int,default=8);p.add_argument('--disassembler',default=shutil.which('tpc-llvm-objdump'));a=p.parse_args()
if a.jobs < 1:p.error('--jobs must be positive')
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
sources={'prefix':'moe_partition/prefix.c','prefix_largest':'moe_partition/prefix.c','inverse':'moe_partition/inverse.c',
 'row_map':'moe_partition/row_map.c','gather':'moe_route_tiles/gather.c',
 'gate':'moe_route_tiles/gate.c','combine':'moe_partition/combine.c','decode':'moe_partition/decode.c','queue':'moe_partition/queue.c','queue_gemv':'moe_partition/masked_gemv_template.c','sparse_map':'moe_partition/sparse_map.c','counted_gather':'moe_partition/counted_gather.c','count':'moe_partition/count.c'}
identity=root/'source-identity.json'
commit=json.loads(identity.read_text())['git_commit'] if identity.exists() and not (root/'.git').exists() else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
names=['csrc/tpc/'+v for v in sources.values()]+['csrc/host/moe_partition_glue.cpp','csrc/torch/moe_partition.cpp','tools/build_moe_partition.py','csrc/tpc/mxfp4_compact/decode_bits.h','csrc/tpc/moe_partition/masked_gemv_baseline.s']
for name in names:
 data=(root/name).read_bytes()
 if identity.exists() and not (root/'.git').exists():assert hashlib.sha256(data).hexdigest()==json.loads(identity.read_text())['files_sha256'][name]
 else:assert data==subprocess.check_output(['git','show',commit+':'+name],cwd=root),'commit sources first'
report=dict(source_commit=commit,commands=[],state='building',device_acquired=False,jobs=a.jobs,
            instruction_evidence='final_disassembly' if a.disassembler else 'compiler_assembly_only')
command_lock=Lock()
def run(cmd,log_name='build.log'):
 command=list(map(str,cmd))
 with command_lock:report['commands'].append(command)
 with (out/log_name).open('a') as log:
  log.write(json.dumps(command)+'\n');log.flush()
  subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
def compile_kernel(item):
  name,source=item
  def kernel_run(cmd):run(cmd,name+'.build.log')
  flags=['-DGK_ROUTE_GATHER_ROW_FIRST=1'] if name=='gather' else []
  if name=='prefix_largest':flags+=['-DGK_EXPERT_LARGEST_M=1']
  if name=='decode':flags+=['-DGK_ROUTE_DECODE_SCALE_BITS=1']
  src=root/'csrc/tpc'/source
  if name=='queue_gemv':
   template=src.read_text().replace('tensor output)','tensor order,tensor output)')
   (out/'queue_template.c').write_text(template);src=out/'queue_template.c';flags+=['-DBLOCKED_LAYOUT=1']
  kernel_run(['tpc-clang','-O2','-mcpu=gaudi2',*flags,'-c',str(src),'-o','pc_'+name+'.o'])
  if name=='queue_gemv':
   s=(root/'csrc/tpc/moe_partition/masked_gemv_baseline.s').read_text()
   kernel_run(['tpc-clang','-mcpu=gaudi2','-c',str(root/'csrc/tpc/moe_partition/masked_gemv_baseline.s'),'-o','baseline_slots.o'])
   kernel_run(['objcopy','--dump-section','.text=baseline.text','baseline_slots.o'])
   assert hashlib.sha256((out/'baseline.text').read_bytes()).hexdigest()=='7c13b008d29b21310a6131bf7954d9de11356c26e7f54e861eeb988e9785e52a'
   # Only task traversal/addressing changes; K32 MAC/reduction ISA stays literal.
   import re
   s=re.sub(r'\bS32\b','S26',s)
   s=s.replace('mov_irf_dim  0x0 S4, I4','ld_l mmio S4, 0x150')
   # Above instruction was in the SPU slot; move MMIO load to LOAD explicitly.
   s=s.replace('nop; \tld_l mmio S4, 0x150; \tnop; \tnop','ld_l mmio S4, 0x150; nop; nop; nop')
   s=s.replace('loop S3, S4, 1, <, .LBB0_7','loop S3, S4, 24, <, .LBB0_7')
   nop='nop; nop; nop; nop\n'
   prefix='set_indx I0, b11111, 0; nop; nop; nop\n'+nop*5+'nop; set_indx I0, b00001, S32; nop; nop\n'+nop*5+'nop; nop; nop; gen_addr AD0, 0x5, I0\n'+nop*6+'ld_g S26, AD0; nop; nop; nop\n'+nop*12
   s=s.replace('.LBB0_2:\n','.LBB0_2:\n'+prefix)
   s=s.replace('st_tnsr 0x5,','st_tnsr 0x6,')
   (out/'queue_gemv.s').write_text(s)
   kernel_run(['tpc-clang','-mcpu=gaudi2','-c','queue_gemv.s','-o','queue_slots.o'])
   kernel_run(['objcopy','--dump-section','.text=queue.text','queue_slots.o'])
   kernel_run(['objcopy','--update-section','.text=queue.text','pc_queue_gemv.o'])
  kernel_run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','pc_'+name+'.o','pc_'+name+'_x86.o'])
  if a.disassembler:
   with (out/(name+'.dis')).open('w') as f:subprocess.run([a.disassembler,'-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',str(out/('pc_'+name+'.o'))],stdout=f,check=True,timeout=30)
  else:
   # Some SDK packages contain the compiler but no TPC disassembler. Preserve
   # compiler ISA and object bytes, without mislabelling source assembly as
   # final binary disassembly. queue_gemv.s above is the literal patched ISA.
   kernel_run(['tpc-clang','-O2','-mcpu=gaudi2',*flags,'-S',str(src),'-o',name+'.compiler.s'])
try:
 with ThreadPoolExecutor(max_workers=min(a.jobs,len(sources))) as pool:list(pool.map(compile_kernel,sources.items()))
 run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-Wl,-z,noexecstack','-I/usr/include/habanalabs',str(root/'csrc/host/moe_partition_glue.cpp'),*['pc_'+n+'_x86.o' for n in sources],'-o','libgaudi_expert_partition_tpc.so'])
 if a.torch:
  os.environ['PT_HPU_LAZY_MODE']='1';os.environ.setdefault('MAX_JOBS','64')
  import habana_frameworks.torch as ht
  from torch.utils.cpp_extension import load
  h=Path(ht.__file__).parent
  load(name='gaudi_expert_partition',sources=[str(root/'csrc/torch/moe_partition.cpp')],extra_include_paths=[str(h/'include'),'/usr/include/habanalabs'],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
 report['state']='built_not_device_verified'
except BaseException as error:
 report.update(state='failed',error=repr(error));raise
finally:
 report['files_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='build.json'}
 (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'state':report['state'],'output':str(out)}))
