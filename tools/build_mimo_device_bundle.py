"""Build the model-owned attention and TP8 TPC kernels as one library."""
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
for group, name in (('full_mme_attention', 'gather'), ('full_mme_attention', 'softmax'),
                    ('tp8_reduce', 'sum_bf16_bf16'), ('tp8_reduce', 'sum_f32_bf16')):
    flags = ['-ffp-contract=off'] if group == 'full_mme_attention' else []
    subprocess.run([args.tpc_clang, '-O2', *flags, '-mcpu=gaudi2', '-c',
                    str(root / f'csrc/tpc/{group}/{name}.c'), '-o', f'{name}.o'],
                   cwd=out, check=True)
    subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                    f'{name}.o', f'{name}_x86.o'], cwd=out, check=True)
    objects.append(f'{name}_x86.o')
for name, prefix in (('full_mme_attention_glue', 'full'), ('tp8_reduce_glue', 'tp8')):
    symbols = ('GetKernelGuids', 'InstantiateTpcKernel', 'GetShapeInference',
               'GetSupportedDataLayout', 'GetLibVersion')
    rename = [f'-D{symbol}={prefix}_{symbol}' for symbol in symbols]
    subprocess.run(['g++', '-O2', '-std=c++17', '-fPIC', f'-I{args.glue_include}',
                    *rename, '-c', str(root / f'csrc/host/{name}.cpp'), '-o', f'{name}.o'],
                   cwd=out, check=True)
    objects.append(f'{name}.o')
subprocess.run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs', '-Wl,-z,noexecstack',
                f'-I{args.glue_include}', str(root / 'csrc/host/mimo_device_bundle_glue.cpp'),
                *objects, '-o', 'libgaudi_model_device_tpc.so'], cwd=out, check=True)
