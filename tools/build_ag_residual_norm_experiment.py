"""Offline-only compile. Torch bridge compilation is opt-in and CPU-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--torch',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
identity=json.loads((root/'source-identity.json').read_text()) if not (root/'.git').exists() else None
commit=identity['git_commit'] if identity else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
paths=['csrc/tpc/ag_residual_norm_experiment/consume.c','csrc/host/ag_residual_norm_experiment_glue.cpp','csrc/torch/ag_residual_norm_experiment.cpp','tools/build_ag_residual_norm_experiment.py']
for name in paths:
    data=(root/name).read_bytes()
    if identity:assert hashlib.sha256(data).hexdigest()==identity['files_sha256'][name],'exported source mismatch'
    else:assert subprocess.check_output(['git','show',commit+':'+name],cwd=root)==data,'commit source first'
    dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
record={'status':'BUILDING','source_commit':commit,'device_used':False,'runtime_qualified':False,'commands':[]}
def run(cmd):
    record['commands'].append(cmd)
    with (out/'build.log').open('a') as log:log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
try:
    for mode,vendor in [('vendor',1),('fp32',0)]:
        for action,suffix in [('-c','o'),('-S','s')]:
            run(['/usr/bin/tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',f'-DVENDOR_BOUNDARIES={vendor}',action,str(root/paths[0]),'-o',f'{mode}.{suffix}'])
        run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',mode+'.o',mode+'_x86.o'])
        with (out/(mode+'.objdump')).open('w') as f:subprocess.run(['tpc-llvm-objdump','--triple=tpc','-d',str(out/(mode+'.o'))],stdout=f,check=True,timeout=30)
    run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',str(root/paths[1]),'vendor_x86.o','fp32_x86.o','-o','libgaudi_ag_norm_experiment_tpc.so'])
    if a.torch:
        os.environ.setdefault('PT_HPU_LAZY_MODE','1');os.environ.setdefault('MAX_JOBS','2')
        import habana_frameworks.torch as ht
        from torch.utils.cpp_extension import load
        h=Path(ht.__file__).parent
        load(name='gaudi_ag_norm_experiment',sources=[str(root/paths[2])],extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
    record['status']='COMPILED_OFFLINE_ONLY'
except Exception as e:record.update(status='FAIL',error=repr(e));raise
finally:
    record['files_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'output':str(out)}))
