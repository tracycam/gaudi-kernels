"""Build independent opt-in reduction GUIDs, retaining the original ELF control."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline-elf',type=Path,required=True);p.add_argument('--include-dir',default='/usr/lib/habanatools/include');p.add_argument('--draft',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);shutil.copy2(a.baseline_elf,out/'reduce_original.o');commands=[]
for chains in [4,8]:
 for flag,ext in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2','-DCHAINS='+str(chains),flag,str(root/'csrc/tpc/block_fp8_reduce_experiment/reduce_chains.c'),'-o',f'reduce_chain{chains}'+ext])
for flag,ext in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',flag,str(root/'csrc/tpc/block_fp8_reduce_experiment/reduce_neumaier.c'),'-o','reduce_neumaier'+ext])
for name in ['reduce_original','reduce_chain4','reduce_chain8','reduce_neumaier']:commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.include_dir,str(root/'csrc/host/block_fp8_reduce_experiment_glue.cpp'),'reduce_original_x86.o','reduce_chain4_x86.o','reduce_chain8_x86.o','reduce_neumaier_x86.o','-o','libblock_fp8_reduce_experiment.so'])
if a.draft:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
else:
 identity=json.loads((root/'source-identity.json').read_text());commit=identity['git_commit']
 for name,want in identity['files_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==want,name
 (out/'source-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
metadata=dict(source_commit=commit,draft=a.draft,baseline_elf_sha256=hashlib.sha256(a.baseline_elf.read_bytes()).hexdigest(),commands=commands,device_validated=False)
try:
 with (out/'build.log').open('w') as log:
  for cmd in commands:log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
 metadata['compiled']=True;metadata['artifacts_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='build.json'}
except Exception as e:metadata.update(compiled=False,error=str(e));raise
finally:(out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
