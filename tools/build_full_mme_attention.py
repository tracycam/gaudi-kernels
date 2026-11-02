"""Build the shared-KV gather and masked-softmax kernels for Gaudi2."""
import argparse
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--tpc-clang', default=shutil.which('tpc-clang') or '/opt/habanalabs/bin/tpc-clang')
parser.add_argument('--glue-include', type=Path, default=Path('/usr/include/habanalabs'))
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
objects = []
for name in ('gather', 'softmax', 'batch_gather', 'batch_softmax'):
    subprocess.run([args.tpc_clang, '-O2', '-ffp-contract=off', '-mcpu=gaudi2', '-c',
                    str(root / f'csrc/tpc/full_mme_attention/{name}.c'), '-o', f'{name}.o'],
                   cwd=out, check=True)
    subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                    f'{name}.o', f'{name}_x86.o'], cwd=out, check=True)
    objects.append(f'{name}_x86.o')
subprocess.run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs', '-Wl,-z,noexecstack',
                f'-I{args.glue_include}', str(root / 'csrc/host/full_mme_attention_glue.cpp'),
                *objects, '-o', 'libgaudi_full_mme_attention_tpc.so'], cwd=out, check=True)
