"""Device-free batch SWA TPC/glue build; retain assembly and executable disassembly."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
names=['head_batch','head_batch_quad']
commands=[]
for name in names:
 source=root/'csrc/tpc/swa128_attention'/f'{name}.c'
 commands.extend([['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',kind,str(source),'-o',name+'.'+suffix]for kind,suffix in (('-c','o'),('-S','s'))])
 commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',str(root/'csrc/host/swa128_batch_glue.cpp'),*[name+'_x86.o'for name in names],'-o','libgaudi_swa128_batch_tpc.so'])
identity=json.loads((root/'source-identity.json').read_text())if(root/'source-identity.json').exists()else {'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
meta=dict(source_commit=identity['git_commit'],commands=commands,status='BUILDING',device_verified=False)
try:
 with(out/'build.log').open('w')as log:
  for cmd in commands:subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
  for name in names:
   with(out/(name+'.dis')).open('w')as dis:
    subprocess.run(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',name+'.o'],cwd=out,stdout=dis,stderr=log,check=True)
 meta['status']='COMPILED'
finally:
 meta['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in out.iterdir()if p.is_file()}
 (out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
