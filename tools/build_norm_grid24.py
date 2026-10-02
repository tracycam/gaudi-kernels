"""Build isolated M1 grid24 public bridge and TPC database without device acquisition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--output-dir',required=True,type=Path)
p.add_argument('--include-dir',default='/usr/include/habanalabs')
p.add_argument('--tpc-compiler',default='tpc-clang')
p.add_argument('--torch-only',action='store_true')
p.add_argument('--tpc-only',action='store_true')
a=p.parse_args()
assert not(a.torch_only and a.tpc_only)
root=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
identity=root/'source-identity.json'
commit=(json.loads(identity.read_text())['git_commit'] if identity.exists() and not (root/'.git').exists()
        else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
meta={'commit':commit,'state':'building','commands':[],'source_sha256':{}}
def source(rel):
    path=root/rel;meta['source_sha256'][rel]=hashlib.sha256(path.read_bytes()).hexdigest();return str(path)
def save(): (out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
save()
try:
    if not a.torch_only:
        variants=[('grid24','csrc/tpc/norm_grid24/main.c',[])]
        objects=[]
        for name,rel,flags in variants:
            path=source(rel)
            for mode,suffix in [('-c','.o'),('-S','.s')]:
                meta['commands'].append([a.tpc_compiler,'-O2','-ffp-contract=off','-mcpu=gaudi2',*flags,mode,path,'-o',name+suffix])
            meta['commands'].append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o']);objects.append(name+'_x86.o')
        meta['commands'].append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.include_dir,source('csrc/host/norm_grid24_glue.cpp'),*objects,'-o','libgaudi_norm_grid24_tpc.so'])
        save()
        with (out/'build.log').open('w') as log:
            for command in meta['commands']:
                log.write(json.dumps(command)+'\n');log.flush();subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not a.tpc_only:
        os.environ.setdefault('PT_HPU_LAZY_MODE','1');os.environ.setdefault('MAX_JOBS','2')
        import torch
        import habana_frameworks.torch as ht
        from torch.utils.cpp_extension import load
        h=Path(ht.__file__).parent
        meta['torch']=torch.__version__;save()
        load(name='gaudi_norm_grid24_torch',sources=[source('csrc/torch/norm_grid24.cpp')],extra_include_paths=[str(h/'include'),a.include_dir],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
    meta['state']='built';meta['artifacts_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='build.json'}
except Exception as error:
    meta.update(state='failed',error=str(error));raise
finally: save()
print(json.dumps({'state':meta['state'],'output_dir':str(out)}))
