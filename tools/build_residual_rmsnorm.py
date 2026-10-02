"""Compile residual RMSNorm TPC plus modern glue; device-free build."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--native',action='store_true')
p.add_argument('--glue-include',default='/usr/lib/habanatools/include');p.add_argument('--tpc-compiler',default='/usr/bin/tpc-clang');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
src=root/'csrc/tpc/residual_rmsnorm/fused.c';glue=root/'csrc/host/residual_rmsnorm_glue.cpp'
commands=[]
objects=[]
for policy,name in enumerate(['fused','per_row','block128']):
 for mode,suffix in [('-c','o'),('-S','s')]:commands.append([a.tpc_compiler,'-O2','-ffp-contract=off','-mcpu=gaudi2',f'-DQUANT_POLICY={policy}',mode,str(src),'-o',f'{name}.{suffix}'])
 commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',f'{name}.o',f'{name}_x86.o']);objects.append(f'{name}_x86.o')
pure=root/'csrc/tpc/residual_rmsnorm/pure.c'
for mode,suffix in [('-c','o'),('-S','s')]:commands.append([a.tpc_compiler,'-O2','-ffp-contract=off','-mcpu=gaudi2',mode,str(pure),'-o',f'pure.{suffix}'])
commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','pure.o','pure_x86.o']);objects.append('pure_x86.o')
commands += [['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.glue_include,str(glue),*objects,'-o','libgaudi_residual_rmsnorm_tpc.so']]
identity=json.loads((root/'source-identity.json').read_text()) if (root/'source-identity.json').exists() else {}
if a.native:commands.append(['g++','-O2','-std=c++17','-I'+a.glue_include,str(root/'benchmarks/residual_rmsnorm/native.cpp'),'-L/usr/lib/habanalabs','-lSynapse','-o','residual_rmsnorm_native'])
m={'commit':identity.get('git_commit') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'status':'exported' if identity else subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),'commands':commands,'state':'building','runtime_verified':False}
for path in [src,pure,glue,root/'benchmarks/residual_rmsnorm/native.cpp',Path(__file__).resolve()]:
 dst=out/'source'/path.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
try:
 with (out/'build.log').open('w') as f:
  for command in commands:f.write(json.dumps(command)+'\n');f.flush();subprocess.run(command,cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True)
 m['state']='compiled'
except Exception as e:m.update(state='failed',error=str(e));raise
finally:
 m['sha256']={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file()};(out/'build.json').write_text(json.dumps(m,indent=2)+'\n')
