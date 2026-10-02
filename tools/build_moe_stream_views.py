"""Build the graph-owned small-M MoE split bridge; no device acquire or launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
os.environ.setdefault('PT_HPU_LAZY_MODE','1');os.environ.setdefault('MAX_JOBS','2')
source=root/'csrc/torch/moe_stream_views.cpp'
identity=root/'source-identity.json'
commit=(json.loads(identity.read_text())['git_commit'] if identity.exists() and not (root/'.git').exists()
        else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
metadata={'commit':commit,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'state':'building','device_acquired':False}
def save():(out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
save()
try:
    commands=[]
    src=root/'csrc/tpc/moe_route_tiles/stream_decode.c'
    for action,suffix in [('-c','o'),('-S','s')]:
        commands.append(['/usr/bin/tpc-clang','-O2','-ffp-contract=off','-DGK_ROUTE_DECODE_SCALE_BITS=1','-mcpu=gaudi2',action,str(src),'-o','stream_decode.'+suffix])
    commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','stream_decode.o','stream_decode_x86.o'])
    commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',str(root/'csrc/host/moe_stream_glue.cpp'),'stream_decode_x86.o','-o','libgaudi_moe_stream_tpc.so'])
    metadata['commands']=commands
    with (out/'tpc-build.log').open('w') as log:
        for cmd in commands:subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    with(out/'stream_decode.dis').open('w') as log:
        subprocess.run(['/usr/bin/tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',str(out/'stream_decode.o')],stdout=log,check=True)
    import torch
    import habana_frameworks.torch as ht
    from torch.utils.cpp_extension import load
    h=Path(ht.__file__).parent;metadata['torch']=torch.__version__
    load(name='gaudi_moe_stream_views',sources=[str(source)],extra_include_paths=[str(h/'include'),'/usr/include/habanalabs'],extra_cflags=['-O2'],
         extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],
         build_directory=str(out),is_python_module=False,verbose=True)
    metadata.update(state='built_not_device_validated',binary_sha256=hashlib.sha256((out/'gaudi_moe_stream_views.so').read_bytes()).hexdigest())
except Exception as error:
    metadata.update(state='failed',error=str(error));raise
finally:save()
