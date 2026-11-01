"""Build the FP32-residual fused SWA normalization kernel without a bridge."""
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
subprocess.run([args.tpc_clang, '-O2', '-ffp-contract=off', '-mcpu=gaudi2', '-c',
                str(root / 'csrc/tpc/norm_grid24_fp32/main.c'), '-o', 'grid24.o'], cwd=out, check=True)
subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                'grid24.o', 'grid24_x86.o'], cwd=out, check=True)
subprocess.run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs', '-Wl,-z,noexecstack',
                f'-I{args.glue_include}', str(root / 'csrc/host/norm_grid24_fp32_glue.cpp'),
                'grid24_x86.o', '-o', 'libgaudi_norm_grid24_fp32_tpc.so'], cwd=out, check=True)
