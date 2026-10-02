"""Actual scheduled assembly and independent graph-local MXFP4 MoE library."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from make_mxfp4_moe import make
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--include',default='/usr/lib/habanatools/include');p.add_argument('--api-include');a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);commands=[];objects=[]
for mode in ['direct','fused','fused256']:
 name='moe_'+mode;make(root/'csrc/tpc/mxfp4_overhead/historical.s',out/(name+'.s'),mode!='direct',mode=='fused256')
 commands.extend([['tpc-clang','-O2','-mcpu=gaudi2','-c',str(root/'csrc/tpc/mxfp4_moe/template.c'),'-o',name+'.o'],['tpc-clang','-mcpu=gaudi2','-c',name+'.s','-o',name+'_slots.o'],['objcopy','--dump-section','.text='+name+'.text',name+'_slots.o'],['objcopy','--update-section','.text='+name+'.text',name+'.o']])
for flag,ext in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',flag,str(root/'csrc/tpc/mxfp4_moe/fused256.c'),'-o','moe_fused256c'+ext])
commands.extend([['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2','-c',str(root/'csrc/tpc/mxfp4_moe/combine.c'),'-o','moe_combine.o'],['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2','-S',str(root/'csrc/tpc/mxfp4_moe/combine.c'),'-o','moe_combine.s']])
legacy=root/'csrc/tpc/mxfp4_moe/legacy'
(out/'moe_legacy_down.s').write_bytes((legacy/'direct_down.s').read_bytes())
commands.extend([['tpc-clang','-O2','-mcpu=gaudi2','-DBLOCKED_LAYOUT=1','-c',str(legacy/'direct_down.c'),'-o','moe_legacy_down.o'],['tpc-clang','-mcpu=gaudi2','-c','moe_legacy_down.s','-o','moe_legacy_down_slots.o'],['objcopy','--dump-section','.text=moe_legacy_down.text','moe_legacy_down_slots.o'],['objcopy','--update-section','.text=moe_legacy_down.text','moe_legacy_down.o']])
for flag,ext in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-mcpu=gaudi2',flag,str(legacy/'combine_f32.c'),'-o','moe_legacy_combine'+ext])
for name in ['moe_direct','moe_fused','moe_combine','moe_fused256','moe_fused256c','moe_legacy_down','moe_legacy_combine']:
 commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o']);objects.append(name+'_x86.o')
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.include,str(root/'csrc/host/mxfp4_moe_glue.cpp'),*objects,'-o','libmxfp4_moe_tpc.so'])
if a.api_include:commands.append(['g++','-O2','-std=c++17','-fPIC','-I'+a.api_include,'-c',str(root/'csrc/ops/mxfp4_moe.cpp'),'-o','mxfp4_moe_graph.o'])
if a.api_include:commands.append(['g++','-O2','-std=c++17','-I'+a.api_include,str(root/'benchmarks/mxfp4_moe/probe.cpp'),'mxfp4_moe_graph.o','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','probe'])
identity=root/'source-identity.json';commit=json.loads(identity.read_text())['git_commit'] if identity.is_file() else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
if identity.is_file():
 doc=json.loads(identity.read_text())
 for name,digest in doc['files_sha256'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('source identity mismatch: '+name)
 (out/'source-identity.json').write_bytes(identity.read_bytes())
meta={'source_commit':commit,'commands':commands,'state':'building'};(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
 with (out/'build.log').open('w') as log:
  for cmd in commands:log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
except Exception as error:meta.update(state='failed',error=str(error));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='compiled_not_validated',artifacts={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='build.json'});(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['state'])
