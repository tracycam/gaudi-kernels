"""Build the imported TPC kernel database only; no device allocation or launch."""
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
source=root/'csrc/tpc/block_fp8'
commands=[];objects=[];sources={}
variants=[('dequant_bf16','dequant_bf16.c',[]),
    ('activation_native','activation_native.c',[]),
    ('reduce1','reduce.c',['-DROWS=1','-DOUTPUT_BF16=0']),
    ('reduce4','reduce.c',['-DROWS=4','-DOUTPUT_BF16=0']),
    ('dequant_native_f32scale','dequant_native.c',['-DSCALE_BF16=0']),
    ('dequant_native_bf16scale','dequant_native_fast.c',['-DSCALE_BF16=1']),
    ('reduce1bf16','reduce.c',['-DROWS=1','-DOUTPUT_BF16=1']),
    ('reduce4bf16','reduce.c',['-DROWS=4','-DOUTPUT_BF16=1'])]
for name,filename,flags in variants:
    path=source/filename;sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for flag,suffix in [('-c','.o'),('-S','.s')]:
        commands.append([args.tpc_compiler,'-O2','-ffp-contract=off','-mcpu=gaudi2',*flags,
            flag,str(path),'-o',name+suffix])
    commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
    objects.append(name+'_x86.o')
glue=root/'csrc/host/block_fp8_glue.cpp'
sources[str(glue.relative_to(root))]=hashlib.sha256(glue.read_bytes()).hexdigest()
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs',
    '-I'+args.include_dir,str(glue),*objects,'-o','libgaudi_block_fp8_tpc.so'])
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
