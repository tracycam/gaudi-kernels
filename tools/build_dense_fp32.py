"""Device-free build for the FP32 Dense kernels."""
import argparse
import hashlib
import json
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
commands = []
objects = []
for stem in ('swiglu', 'residual_rmsnorm', 'linear'):
    source = root / f'csrc/tpc/dense_fp32/{stem}.c'
    commands.append([args.tpc_clang, '-O2', '-ffp-contract=off', '-mcpu=gaudi2', '-c',
                     str(source), '-o', f'{stem}.o'])
    commands.append([args.tpc_clang, '-O2', '-ffp-contract=off', '-mcpu=gaudi2', '-S',
                     str(source), '-o', f'{stem}.s'])
    commands.append(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                     f'{stem}.o', f'{stem}_x86.o'])
    objects.append(f'{stem}_x86.o')
commands.append(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs', '-Wl,-z,noexecstack',
                 f'-I{args.glue_include}', str(root / 'csrc/host/dense_fp32_glue.cpp'),
                 *objects, '-o', 'libgaudi_dense_fp32_tpc.so'])
metadata = {'commands': commands, 'status': 'BUILDING', 'device_verified': False}
for name in ('csrc/tpc/dense_fp32/swiglu.c',
             'csrc/tpc/dense_fp32/residual_rmsnorm.c', 'csrc/tpc/dense_fp32/linear.c',
             'csrc/host/dense_fp32_glue.cpp', 'tools/build_dense_fp32.py'):
    destination = out / 'source' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / name, destination)
try:
    with (out / 'build.log').open('w') as log:
        for command in commands:
            log.write(json.dumps(command) + '\n')
            log.flush()
            subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata['status'] = 'COMPILED'
finally:
    metadata['files_sha256'] = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in out.rglob('*') if path.is_file()}
    (out / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
