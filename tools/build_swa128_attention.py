"""Device-free Gaudi2 TPC and glue build for the explicitly experimental route."""
import argparse,hashlib,json,subprocess,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
commands=[];objects=[]
for stem in ['head','head_fp32','head_window','head_fast','head_window_fast']:
 for kind,suffix in [('-c','o'),('-S','s')]:commands.append(['/usr/bin/tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',kind,str(root/f'csrc/tpc/swa128_attention/{stem}.c'),'-o',f'{stem}.{suffix}'])
 commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',f'{stem}.o',f'{stem}_x86.o']);objects.append(f'{stem}_x86.o')
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',str(root/'csrc/host/swa128_attention_glue.cpp'),*objects,'-o','libgaudi_swa128_attention_tpc.so'])
metadata={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'commands':commands,'device_verified':False,'vendor_numerics_verified':False,'status':'BUILDING'}
for name in ['csrc/tpc/swa128_attention/head.c','csrc/tpc/swa128_attention/head_fp32.c','csrc/tpc/swa128_attention/head_window.c','csrc/tpc/swa128_attention/head_fast.c','csrc/tpc/swa128_attention/head_window_fast.c','csrc/host/swa128_attention_glue.cpp','tools/build_swa128_attention.py']:
 dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dst)
try:
 with (out/'build.log').open('w') as log:
  for c in commands:log.write(json.dumps(c)+'\n');log.flush();subprocess.run(c,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
 metadata['status']='COMPILED'
finally:
 metadata['files_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()};(out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
