"""Device-free build for the multi-page DiffKV attention kernels."""
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
for stem in ('block', 'reduce'):
    for kind, suffix in (('-c', 'o'), ('-S', 's')):
        commands.append([args.tpc_clang, '-O2', '-ffp-contract=off', '-mcpu=gaudi2', kind,
                         str(root / f'csrc/tpc/full_diffkv_attention/{stem}.c'), '-o', f'{stem}.{suffix}'])
    commands.append(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                     f'{stem}.o', f'{stem}_x86.o'])
    objects.append(f'{stem}_x86.o')
commands.append(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs', '-Wl,-z,noexecstack',
                 f'-I{args.glue_include}', str(root / 'csrc/host/full_diffkv_attention_glue.cpp'),
                 *objects, '-o', 'libgaudi_full_diffkv_attention_tpc.so'])
revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                          capture_output=True, check=False)
metadata = {'source_commit': revision.stdout.strip() if revision.returncode == 0 else None,
            'commands': commands, 'device_verified': False, 'status': 'BUILDING'}
for name in ('csrc/tpc/full_diffkv_attention/block.c', 'csrc/tpc/full_diffkv_attention/reduce.c',
             'csrc/host/full_diffkv_attention_glue.cpp', 'tools/build_full_diffkv_attention.py'):
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
