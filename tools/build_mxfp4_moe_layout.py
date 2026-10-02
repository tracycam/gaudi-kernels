"""Offline audited legacy ASM and real shared-owner decoder experiment."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);commands=[];objects=[]
for name in ['native','historical']:
 for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',*(['-DGK_HISTORICAL_N512=1'] if name=='historical' else []),flag,str(root/'csrc/tpc/mxfp4_moe/bucket_decode.c'),'-o','layout_'+name+suffix])
for name in ['gp','down']:
 base=root/'csrc/tpc/mxfp4_moe/legacy';(out/('layout_'+name+'.s')).write_bytes((base/('direct_'+name+'.s')).read_bytes())
 commands += [['tpc-clang','-O2','-mcpu=gaudi2','-DBLOCKED_LAYOUT=1','-c',str(base/('direct_'+name+'.c')),'-o','layout_'+name+'.o'],['tpc-clang','-mcpu=gaudi2','-c','layout_'+name+'.s','-o','layout_'+name+'_asm.o'],['objcopy','--dump-section','.text=layout_'+name+'.text','layout_'+name+'_asm.o'],['objcopy','--update-section','.text=layout_'+name+'.text','layout_'+name+'.o']]
for name in ['native','historical','gp','down']:
 commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','layout_'+name+'.o','layout_'+name+'_x86.o']);objects.append('layout_'+name+'_x86.o')
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',str(root/'benchmarks/mxfp4_moe_layout/glue.cpp'),*objects,'-o','libmxfp4_layout_tpc.so'])
commands.append(['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(root/'benchmarks/mxfp4_moe_layout/audit.cpp'),'-L'+str(out),'-lmxfp4_layout_tpc','-L/usr/lib/habanatools','-ltpc_tests_core_ext','-Wl,-rpath,'+str(out)+':/usr/lib/habanatools','-o','audit'])
meta={'commands':commands,'device_validated':False,'state':'building','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'working_tree_diff':subprocess.check_output(['git','diff','--stat'],cwd=root,text=True)};(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
 with (out/'build.log').open('w') as log:
  for cmd in commands:log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
except Exception as ex:meta.update(state='failed',error=str(ex));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='offline_compiled',artifacts={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='build.json'});(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
