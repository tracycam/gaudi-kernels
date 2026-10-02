"""Offline-first exact MXFP4 build; compilation never means device validation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from make_mxfp4_conditional import make

p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--abi-include',type=Path,help='Current tpc_kernel_lib_interface.h directory, optional offline')
p.add_argument('--api-include',type=Path,help='Complete current Synapse public headers, optional offline')
a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
source=root/'csrc/tpc/mxfp4_overhead/exact.c';commands=[];objects=[]
for compact in (0,1):
 for repair in (0,1):
  for bias in (0,1):
   name='mx_'+('repair' if repair else 'exact')+('_v3' if compact else '_v2')+('_bias' if bias else '')
   for flag,suffix in [('-c','.o'),('-S','.s')]:
    commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',f'-DCOMPACT={compact}',f'-DREPAIR={repair}',f'-DBIAS={bias}',flag,str(source),'-o',name+suffix])
   commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o']);objects.append(name+'_x86.o')
family=source.parent
make(family/'historical.s',out/'mx_dispatch_history.s')
template=(family/'historical_template.c').read_text().replace('tensor mapping,tensor output','tensor mapping,tensor flags,tensor output')
template=template.replace('int5 mi={block,0,0,0,0};','if(s_u16_ld_g(gen_addr((int5){0,block,0,0,0},flags)))continue;\n        int5 mi={block,0,0,0,0};')
(out/'mx_dispatch_template.c').write_text(template)
for name,file,flags in [('mx_dispatch_prepare',family/'predispatch_prepare.c',[]),('mx_dispatch_prepare_bias',family/'predispatch_prepare.c',['-DBIAS=1']),('mx_dispatch_finish',source,['-DDISPATCH=1']),('mx_dispatch_finish_bias',source,['-DDISPATCH=1','-DBIAS=1'])]:
 for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',*flags,flag,str(file),'-o',name+suffix])
 commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o']);objects.append(name+'_x86.o')
commands.extend([
 ['tpc-clang','-O2','-mcpu=gaudi2','-DBLOCKED_LAYOUT=1','-c','mx_dispatch_template.c','-o','mx_dispatch_history.o'],
 ['tpc-clang','-mcpu=gaudi2','-c','mx_dispatch_history.s','-o','mx_dispatch_slots.o'],
 ['objcopy','--dump-section','.text=mx_dispatch_history.text','mx_dispatch_slots.o'],
 ['objcopy','--update-section','.text=mx_dispatch_history.text','mx_dispatch_history.o'],
 ['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','mx_dispatch_history.o','mx_dispatch_history_x86.o'],
 ['g++','-O2','-std=c++17','-shared','-fPIC',str(root/'benchmarks/mxfp4_exact/native.cpp'),'-o','native.so']])
objects.append('mx_dispatch_history_x86.o')
if a.abi_include:
 commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+str(a.abi_include.resolve()),str(root/'csrc/host/mxfp4_exact_glue.cpp'),*objects,'-o','libmxfp4_exact_tpc.so'])
if a.api_include:
 commands.append(['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-fPIC','-I'+str(a.api_include.resolve()),'-c',str(root/'csrc/ops/mxfp4_exact.cpp'),'-o','mxfp4_exact_graph.o'])
paths=[source,*source.parent.glob('exact_*.h'),family/'predispatch_prepare.c',family/'historical.s',family/'historical_template.c',root/'tools/make_mxfp4_conditional.py',root/'csrc/host/mxfp4_exact_glue.cpp',*root.glob('csrc/ops/mxfp4_exact.*'),*root.glob('benchmarks/mxfp4_exact/*'),Path(__file__)]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
identity_path=root/'source-identity.json'
if identity_path.is_file():
 identity=json.loads(identity_path.read_text());commit=identity['git_commit']
 for name,digest in identity['files_sha256'].items():
  path=root/name
  if not path.is_file() or sha(path)!=digest:raise RuntimeError('source identity mismatch: '+name)
else:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
meta={'git_commit':commit,'source_sha256':{str(f.relative_to(root)):sha(f) for f in paths if f.is_file()},'commands':commands,'state':'building','device_validated':False,'glue_requested':bool(a.abi_include),'graph_core_requested':bool(a.api_include)}
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
 with (out/'build.log').open('w') as log:
  for command in commands:
   log.write(json.dumps(command)+'\n');log.flush();subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
except Exception as error:
 meta.update(state='failed',error=str(error));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='offline_compiled_not_device_validated',artifacts={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='build.json'})
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'state':meta['state'],'output':str(out)}))
