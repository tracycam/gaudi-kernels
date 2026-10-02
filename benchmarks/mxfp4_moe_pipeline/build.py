"""Build the counted GP/SiLU/down/combine graph experiment. No acquisition."""
import argparse,json,subprocess,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
src=Path(__file__).resolve().parent;out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);commands=[]
for name in ('gate','combine'):
 for flag,suffix in (('-c','.o'),('-S','.s')):commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',flag,str(src/(name+'.c')),'-o',name+suffix])
 commands += [['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'],['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',name+'.o']]
commands += [['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/include/habanalabs',str(src/'glue.cpp'),'gate_x86.o','combine_x86.o','-o','libchain.so'],['g++','-O3','-std=c++17','-ffp-contract=off','-I/usr/include/habanalabs',str(src/'probe.cpp'),'-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','probe']]
record=dict(state='building',commands=commands)
try:
 with (out/'build.log').open('w')as log:
  for cmd in commands:
   log.write(json.dumps(cmd)+'\n');log.flush()
   if cmd[0]=='tpc-llvm-objdump':
    with (out/(cmd[-1]+'.dis')).open('w')as f:subprocess.run(cmd,cwd=out,stdout=f,stderr=log,check=True)
   else:subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
 record['state']='built'
finally:
 record['sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in out.iterdir()if f.is_file()and f.name!='build.json'};(out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(state=record['state'],output=str(out))))
