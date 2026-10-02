"""Build actual historical assembly plus guarded prepared-layout candidates."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
src=root/'csrc/tpc/mxfp4_overhead';commands=[];objects=[]
history=(src/'historical.s').read_text();(out/'oh_history.s').write_text(history)
prologue=history.replace('cmp_grt.i32  SP1, S0, 0x1f','cmp_eq.i32  SP1, S0, 0xffffffff')
assert prologue!=history;(out/'oh_prologue.s').write_text(prologue)
for name in ['history','prologue']:
 target='oh_'+name
 commands.extend([['tpc-clang','-O2','-mcpu=gaudi2','-DBLOCKED_LAYOUT=1','-c',str(src/'historical_template.c'),'-o',target+'.o'],
 ['tpc-clang','-mcpu=gaudi2','-c',target+'.s','-o',target+'_slots.o'],
 ['objcopy','--dump-section','.text='+target+'.text',target+'_slots.o'],
 ['objcopy','--update-section','.text='+target+'.text',target+'.o']])
for name,file,flags in [('guarded','guarded.c',[]),('exact','guarded.c',['-DEXACT=1']),('prepare','prepare.c',[]),('finish','finish.c',[]),('finish_bias','finish.c',['-DBIAS=1'])]:
 for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',*flags,flag,str(src/file),'-o','oh_'+name+suffix])
for name in ['history','guarded','exact','prepare','finish','finish_bias','prologue']:
 target='oh_'+name;commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',target+'.o',target+'_x86.o']);objects.append(target+'_x86.o')
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/include/habanalabs',str(root/'csrc/host/mxfp4_overhead_glue.cpp'),*objects,'-o','libmxfp4_overhead_tpc.so'])
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/include/habanalabs',str(root/'csrc/ops/mxfp4_overhead.cpp'),'-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','libmxfp4_overhead.so'])
commands.append(['python3',str(root/'tools/build_mxfp4_linear.py'),'--output-dir',str(out/'baseline')])
commands.append(['g++','-O3','-std=c++17','-I/usr/include/habanalabs',str(root/'benchmarks/mxfp4_overhead/probe.cpp'),'-L.','-Lbaseline','-Wl,-rpath,$ORIGIN:$ORIGIN/baseline','-lmxfp4_overhead','-lgaudi_mxfp4_linear','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','overhead_probe'])
paths=[*src.glob('*'),root/'csrc/host/mxfp4_overhead_glue.cpp',*root.glob('csrc/ops/mxfp4_overhead.*'),*root.glob('benchmarks/mxfp4_overhead/*'),Path(__file__)]
meta={'source_sha256':{str(s.relative_to(root)):hashlib.sha256(s.read_bytes()).hexdigest() for s in paths if s.is_file()},'commands':commands,'state':'building'}
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
with (out/'build.log').open('w') as log:
 try:
  for command in commands:log.write(json.dumps(command)+'\n');log.flush();subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
 except Exception as e:
  meta.update(state='failed',error=str(e));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='built_not_validated',artifacts={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='build.json'})
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'state':meta['state'],'output':str(out)}))
