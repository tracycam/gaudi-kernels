"""Build endpoint probes, ELF and disassembly. Does not acquire a device."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--write-unroll',type=int,choices=[8,32],default=8);a=p.parse_args()
src=Path(__file__).resolve().parent;out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
commands=[]
names=('read','write','copy','read_il','write_il','copy_il','drain','write_flat')
for i,name in enumerate(names):
    for option,suffix in (('-c','.o'),('-S','.s')):
        commands.append(['tpc-clang','-O2','-mcpu=gaudi2',f'-DPORT_MODE={i%3}',f'-DPORT_WRITE_UNROLL={a.write_unroll}',*(['-DPORT_INTERLEAVE=1'] if 3<=i<6 else []),*(['-DPORT_FLAT=1'] if i==7 else []),option,str(src/('drain.c' if i==6 else 'traffic.c')),'-o',name+suffix])
    commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
    commands.append(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',name+'.o'])
commands.append(['g++','-O2','-std=c++17',f'-DPORT_WRITE_UNROLL={a.write_unroll}','-shared','-fPIC','-Wl,-z,defs','-I/usr/include/habanalabs',str(src/'glue.cpp'),*[n+'_x86.o' for n in names],'-o','libports.so'])
commands.append(['g++','-O3','-std=c++17',f'-DPORT_WRITE_UNROLL={a.write_unroll}','-I/usr/include/habanalabs',str(src/'probe.cpp'),'-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','probe'])
record=dict(state='building',commands=commands)
try:
    with (out/'build.log').open('w') as log:
        for cmd in commands:
            log.write(json.dumps(cmd)+'\n');log.flush()
            if cmd[0]=='tpc-llvm-objdump':
                with (out/(cmd[-1]+'.dis')).open('w') as dis:subprocess.run(cmd,cwd=out,stdout=dis,stderr=log,check=True)
            else:subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    record['state']='built'
finally:
    record['sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name!='build.json'}
    (out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(state=record['state'],output=str(out))))
