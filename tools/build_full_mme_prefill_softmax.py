"""Build the standalone M32 causal full-attention softmax provider."""
import argparse
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--tpc-clang', default=shutil.which('tpc-clang') or '/opt/habanalabs/bin/tpc-clang')
p.add_argument('--glue-include', type=Path, default=Path('/usr/include/habanalabs'))
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
source = root / 'csrc/tpc/full_mme_attention/prefill_softmax.c'
glue = root / 'csrc/host/full_mme_prefill_softmax_glue.cpp'
subprocess.run([a.tpc_clang, '-O2', '-ffp-contract=off', '-mcpu=gaudi2', '-c',
                str(source), '-o', 'prefill_softmax.o'], cwd=out, check=True)
subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                'prefill_softmax.o', 'prefill_softmax_x86.o'], cwd=out, check=True)
subprocess.run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs',
                '-Wl,-z,noexecstack', f'-I{a.glue_include}', str(glue),
                'prefill_softmax_x86.o', '-o', 'libgaudi_full_mme_prefill_softmax_tpc.so'],
               cwd=out, check=True)
