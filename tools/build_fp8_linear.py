"""Build ordinary FP8 TPC kernels and standalone benchmark; no device allocation or launch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',required=True,type=Path)
parser.add_argument('--include-dir',default='/usr/include/habanalabs')
parser.add_argument('--tpc-compiler',default='tpc-clang')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
out=args.output_dir.resolve()
out.mkdir(parents=True,exist_ok=False)
source=root/'csrc/tpc/fp8_linear'
commands=[];objects=[];sources={}
variants=[('activation','activation.c',[]),('activation_fast','activation_fast.c',[]),('decode','decode.c',[]),('decode128','decode.c',['-DROW_BLOCK=128']),
    ('epilogue8_rows','epilogue_rows.c',['-DA8=1']),('epilogue16_rows','epilogue_rows.c',['-DA8=0']),
    ('epilogue8_fast','epilogue_fast.c',['-DA8=1']),('epilogue16_fast','epilogue_fast.c',['-DA8=0']),
    ('epilogue8','epilogue.c',['-DA8=1']),('epilogue16','epilogue.c',['-DA8=0'])]
for name,filename,flags in variants:
    path=source/filename;sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for flag,suffix in [('-c','.o'),('-S','.s')]:
        commands.append([args.tpc_compiler,'-O2','-ffp-contract=off','-mcpu=gaudi2',*flags,
            flag,str(path),'-o',name+suffix])
    commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
    objects.append(name+'_x86.o')
glue=root/'csrc/host/fp8_linear_glue.cpp'
sources[str(glue.relative_to(root))]=hashlib.sha256(glue.read_bytes()).hexdigest()
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs',
    '-I'+args.include_dir,str(glue),*objects,'-o','libgaudi_fp8_linear_tpc.so'])
core=root/'csrc/ops/fp8_linear.cpp'
bench=root/'benchmarks/fp8_linear/bench.cpp'
for path in [core,bench,root/'csrc/ops/fp8_linear.hpp',root/'csrc/common/experiment_device.hpp']:
    sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
commands.append(['g++','-O2','-std=c++17','-I'+args.include_dir,'-I'+str(root/'csrc'),
    str(core),str(bench),'-L/usr/lib/habanalabs','-lSynapse','-o','fp8_linear_bench'])
metadata={'source_sha256':sources,'commands':commands,'state':'building',
    'scope':'TPC kernel database; no PyTorch OpBackend binding, no new HPU validation'}
(out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
with (out/'build.log').open('w') as log:
    try:
        for command in commands:
            log.write(json.dumps(command)+'\n');log.flush()
            subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    except (OSError,subprocess.CalledProcessError) as error:
        metadata['state']='failed'
        metadata['error']=str(error)
        (out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
        raise
metadata['state']='built_not_device_retested'
metadata['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(out.iterdir()) if p.is_file() and p.name!='build.json'}
(out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps({'state':metadata['state'],'output_dir':str(out)}))
