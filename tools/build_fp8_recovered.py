"""Build both experimental/previous FP8 libraries and native device probes; no acquire."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path)
p.add_argument('--include-dir',default='/usr/include/habanalabs');p.add_argument('--tpc-compiler',default='/usr/bin/tpc-clang')
a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
if (root/'source-identity.json').is_file():identity=json.loads((root/'source-identity.json').read_text());commit=identity['git_commit']
else:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
commands=[];sources={};objects={'linear':[],'first_principles':[]}
def source(path):
 path=root/path;rel=path.relative_to(root);target=out/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
 sources[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest();return path
variants=[('activation','activation.c',[]),('activation_fast','activation_fast.c',[]),('decode','decode.c',[]),('decode128','decode.c',['-DROW_BLOCK=128']),('epilogue8','epilogue.c',['-DA8=1']),('epilogue16','epilogue.c',['-DA8=0']),('epilogue8_fast','epilogue_fast.c',['-DA8=1']),('epilogue16_fast','epilogue_fast.c',['-DA8=0']),('epilogue8_rows','epilogue_rows.c',['-DA8=1']),('epilogue16_rows','epilogue_rows.c',['-DA8=0'])]
variants=[('linear',name,'csrc/tpc/fp8_linear/'+file,flags) for name,file,flags in variants]
variants += [('first_principles',name,'csrc/tpc/fp8_first_principles/quantize.c',[f'-DUSE_LUT={lut}',f'-DCORRECT_QUOTIENT={correction}',f'-DAMAX_CHAINS={chains}']) for name,lut,correction,chains in [('amax16',0,1,1),('lut_corrected',1,1,1),('lut_single',1,0,1),('lut_single4',1,0,4)]]
for family,name,path,flags in variants:
 path=source(path)
 for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append([a.tpc_compiler,'-O2','-ffp-contract=off','-mcpu=gaudi2',*flags,flag,str(path),'-o',name+suffix])
 commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o']);objects[family].append(name+'_x86.o')
for family in objects:
 glue=source('csrc/host/fp8_'+family+'_glue.cpp');commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.include_dir,str(glue),*objects[family],'-o','libgaudi_fp8_'+family+'_tpc.so'])
core=source('csrc/ops/fp8_linear.cpp');source('csrc/ops/fp8_linear.hpp');source('csrc/common/experiment_device.hpp')
for name,path,extra in [('fp8_quant_bench','benchmarks/fp8_first_principles/device_quant.cpp',[]),('fp8_linear_bench','benchmarks/fp8_linear/bench.cpp',[str(core)])]:
 bench=source(path);commands.append(['g++','-O2','-std=c++17','-I'+a.include_dir,'-I'+str(root/'csrc'),str(bench),*extra,'-L/usr/lib/habanalabs','-lSynapse','-o',name])
source('tools/build_fp8_recovered.py')
meta={'state':'building','source_commit':commit,'source_sha256':sources,'commands':commands,'compiler_version':subprocess.check_output([a.tpc_compiler,'--version'],text=True),'compiler_sha256':hashlib.sha256(Path(shutil.which(a.tpc_compiler)).read_bytes()).hexdigest()}
manifest=out/'build.json';manifest.write_text(json.dumps(meta,indent=2)+'\n')
try:
 with (out/'build.log').open('w') as log:
  for cmd in commands:log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
except (OSError,subprocess.CalledProcessError) as error:meta.update(state='failed',error=str(error));manifest.write_text(json.dumps(meta,indent=2)+'\n');raise
meta['state']='built_not_device_validated';meta['artifacts']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p!=manifest};manifest.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'state':meta['state'],'output_dir':str(out)}))
