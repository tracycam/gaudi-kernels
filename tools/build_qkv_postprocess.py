"""Offline compile of four explicit arithmetic candidates; no device access."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--glue-include',default='/usr/lib/habanatools/include');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);src=root/'csrc/tpc/qkv_postprocess/postprocess.c';glue=root/'csrc/host/qkv_postprocess_glue.cpp';commands=[];objects=[]
for ri,r in enumerate(['bf16','f32']):
 for vi,v in enumerate(['bf16','f32']):
  name=r+'_'+v
  for flag,suffix in [('-c','o'),('-S','s')]:commands.append(['/usr/bin/tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',f'-DROPE_F32={ri}',f'-DVSCALE_F32={vi}',flag,str(src),'-o',name+'.'+suffix])
  commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o']);objects.append(name+'_x86.o')
for flag,suffix in [('-c','o'),('-S','s')]:commands.append(['/usr/bin/tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2','-DCACHE_INPUT=1',flag,str(src),'-o','cache_bf16_v2.'+suffix])
commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','cache_bf16_v2.o','cache_bf16_v2_x86.o']);objects.append('cache_bf16_v2_x86.o')
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.glue_include,str(glue),*objects,'-o','libgaudi_qkv_postprocess_tpc.so'])
meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'commands':commands,'state':'building','device_verified':False}
for path in [src,glue,root/'csrc/torch/qkv_postprocess.cpp',root/'python/gaudi_kernels/qkv_postprocess.py',Path(__file__).resolve()]:
 target=out/'source'/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
try:
 with (out/'build.log').open('w') as log:
  for command in commands:log.write(json.dumps(command)+'\n');log.flush();subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
 meta['state']='compiled_offline'
except Exception as e:meta.update(state='failed',error=str(e));raise
finally:
 meta['sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()};(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
