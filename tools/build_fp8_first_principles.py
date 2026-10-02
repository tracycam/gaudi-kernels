"""Offline TPC/host build; never initializes a runtime or accesses a device."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--output-dir',required=True,type=Path)
p.add_argument('--tpc-compiler',default='/usr/bin/tpc-clang')
p.add_argument('--glue-include',default='/usr/lib/habanatools/include')
p.add_argument('--synapse-include',type=Path)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
synapse=a.synapse_include
if synapse is None:
    options=[base/'deps/SynapseAI_Core/external_includes' for base in [root.parent,root.parent.parent]]
    synapse=next((path for path in options if (path/'synapse_api.h').is_file()),None)
    if synapse is None:p.error('cannot locate synapse_api.h; pass --synapse-include')
out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
source=root/'csrc/tpc/fp8_first_principles/quantize.c'
glue=root/'csrc/host/fp8_first_principles_glue.cpp'
fragment=root/'csrc/ops/fp8_first_principles.cpp'
paths=[source,glue,fragment,fragment.with_suffix('.hpp'),Path(__file__).resolve(),
       root/'benchmarks/fp8_first_principles/cpu_gate.py',root/'benchmarks/fp8_first_principles/isa_cost.py',
       root/'csrc/tpc/fp8_linear/activation_fast.c']
source_hashes={}
for path in paths:
    rel=path.relative_to(root);target=out/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target);source_hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
commands=[];objects=[]
for name,lut,correction,chains in [('amax16',0,1,1),('lut_corrected',1,1,1),('lut_single',1,0,1),('lut_single4',1,0,4),('baseline',0,1,1)]:
    path=source if name!='baseline' else root/'csrc/tpc/fp8_linear/activation_fast.c'
    for flag,suffix in [('-c','.o'),('-S','.s')]:
        commands.append([a.tpc_compiler,'-O2','-ffp-contract=off','-mcpu=gaudi2',
                         f'-DUSE_LUT={lut}',f'-DCORRECT_QUOTIENT={correction}',f'-DAMAX_CHAINS={chains}',flag,str(path),'-o',name+suffix])
    if name!='baseline':
        commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
        objects.append(name+'_x86.o')
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs',
                 '-I'+a.glue_include,str(glue),*objects,'-o','libgaudi_fp8_first_principles_tpc.so'])
commands.append(['g++','-O2','-std=c++17','-I'+str(synapse),'-c',str(fragment),'-o','graph_fragment.o'])
metadata={'state':'building','runtime_verified':False,'source_sha256':source_hashes,'commands':commands,
          'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
          'worktree_status':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),
          'compiler_version':subprocess.check_output([a.tpc_compiler,'--version'],text=True),
          'compiler_sha256':hashlib.sha256(Path(shutil.which(a.tpc_compiler)).read_bytes()).hexdigest()}
manifest=out/'build.json';manifest.write_text(json.dumps(metadata,indent=2)+'\n')
try:
    with (out/'build.log').open('w') as log:
        for command in commands:
            log.write(json.dumps(command)+'\n');log.flush()
            subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
except (OSError,subprocess.CalledProcessError) as error:
    metadata.update(state='failed',error=str(error));manifest.write_text(json.dumps(metadata,indent=2)+'\n');raise
metadata['state']='compiled_offline_runtime_pending'
metadata['artifacts']={str(path.relative_to(out)):hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(out.rglob('*')) if path.is_file() and path!=manifest}
manifest.write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({'state':metadata['state'],'output_dir':str(out)}))
