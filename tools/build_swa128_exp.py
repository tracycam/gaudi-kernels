"""Build the isolated M=1 SWA exponential-correction provider."""
import argparse
import shutil
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--tpc-clang', default=shutil.which('tpc-clang') or '/opt/habanalabs/bin/tpc-clang')
p.add_argument('--glue-include', default='/usr/include/habanalabs')
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
subprocess.run([args.tpc_clang, '-O2', '-ffp-contract=off', '-mcpu=gaudi2', '-c',
                str(root / 'csrc/tpc/swa128_ilp/head_quad.c'), '-o', 'swa.o'], cwd=out, check=True)
subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                'swa.o', 'swa_x86.o'], cwd=out, check=True)
subprocess.run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs', '-Wl,-z,noexecstack',
                '-I' + args.glue_include, str(root / 'csrc/host/swa128_exp_glue.cpp'),
                'swa_x86.o', '-o', 'libgaudi_swa128_exp_tpc.so'], cwd=out, check=True)
