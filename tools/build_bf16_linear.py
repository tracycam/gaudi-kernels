"""Build BF16 public-Synapse graph core, benchmark and TPC library; no device use."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--include-dir',default='/usr/include/habanalabs');p.add_argument('--tpc-compiler',default='tpc-clang');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
source=root/'csrc/tpc/bf16_linear';commands=[];objects=[]
variants=[('gemv_acc321','gemv.c',['-DROWS=1','-DNATIVE_MAC=1']),('gemv_acc322','gemv.c',['-DROWS=2','-DNATIVE_MAC=1']),('gemv1','gemv.c',['-DROWS=1']),('gemv2','gemv.c',['-DROWS=2'])]
for reduce,bias in [(1,0),(1,1),(0,1)]:
 for bf16 in [1,0]:
  name=('reduce' if reduce else 'bias')+('_bias' if reduce and bias else '')+('_bf16' if bf16 else '_f32')
  variants.append((name,'epilogue.c',[f'-DREDUCE={reduce}',f'-DBIAS={bias}',f'-DOUTPUT_BF16={bf16}']))
for unroll in [0,1]:
 for bias in [0,1]:
  for bf16 in [1,0]:
   name=('rowdot4' if unroll else 'rowdot')+('_bias' if bias else '')+('_bf16' if bf16 else '_f32')
   variants.append((name,'row_dot.c',[f'-DBIAS={bias}',f'-DOUTPUT_BF16={bf16}',f'-DUNROLL4={unroll}']))
for name,filename,flags in variants:
 for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append([a.tpc_compiler,'-O2','-ffp-contract=off','-mcpu=gaudi2',*flags,flag,str(source/filename),'-o',name+suffix])
 commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o']);objects.append(name+'_x86.o')
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.include_dir,str(root/'csrc/host/bf16_linear_glue.cpp'),*objects,'-o','libgaudi_bf16_linear_tpc.so'])
commands.append(['g++','-O2','-std=c++17','-fPIC','-I'+a.include_dir,'-c',str(root/'csrc/ops/bf16_linear.cpp'),'-o','bf16_linear.o'])
commands.append(['ar','rcs','libgaudi_bf16_linear.a','bf16_linear.o'])
commands.append(['g++','-O2','-std=c++17','-I'+a.include_dir,str(root/'benchmarks/bf16_linear/native.cpp'),'libgaudi_bf16_linear.a','-L/usr/lib/habanalabs','-lSynapse','-o','bf16_linear_benchmark'])
files=[*source.glob('*.c'),root/'csrc/host/bf16_linear_glue.cpp',root/'csrc/ops/bf16_linear.cpp',root/'csrc/ops/bf16_linear.hpp',root/'benchmarks/bf16_linear/native.cpp',Path(__file__).resolve()]
metadata={'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'commands':commands,'state':'building'}
(out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
with (out/'build.log').open('w') as log:
 try:
  for command in commands:
   log.write(json.dumps(command)+'\n');log.flush();subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
 except (OSError,subprocess.CalledProcessError) as e:
  metadata.update(state='failed',error=str(e));(out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n');raise
metadata['state']='built';metadata['artifacts']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir()) if f.is_file() and f.name!='build.json'}
(out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({'state':'built','output_dir':str(out)}))
