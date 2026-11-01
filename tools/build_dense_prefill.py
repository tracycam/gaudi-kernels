"""Build the M32 FP32 dense prefill provider."""
import argparse
from pathlib import Path
import shutil
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--tpc-clang',default=shutil.which('tpc-clang') or '/opt/habanalabs/bin/tpc-clang')
p.add_argument('--glue-include',type=Path,default=Path('/usr/include/habanalabs'))
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
objects=[]
for name,source in [('swiglu','csrc/tpc/dense_prefill/swiglu.c'),
                    ('residual_rmsnorm','csrc/tpc/dense_fp32/residual_rmsnorm.c'),
                    ('linear','csrc/tpc/dense_prefill/linear.c')]:
    subprocess.run([a.tpc_clang,'-O2','-ffp-contract=off','-mcpu=gaudi2','-c',str(root/source),'-o',name+'.o'],cwd=out,check=True)
    subprocess.run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'],cwd=out,check=True)
    objects.append(name+'_x86.o')
subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-Wl,-z,noexecstack',
                f'-I{a.glue_include}',str(root/'csrc/host/dense_prefill_glue.cpp'),*objects,
                '-o','libgaudi_dense_prefill_tpc.so'],cwd=out,check=True)
