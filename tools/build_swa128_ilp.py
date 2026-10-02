"""Build isolated ILP experiments; no production library replacement or card."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
sources = ['csrc/tpc/swa128_attention/head_fp32.c',
           'csrc/tpc/swa128_attention/head_window_fast.c',
           'benchmarks/swa128_attention/oracle.py',
           'tools/build_swa128_ilp.py']
sources += [str(p.relative_to(root)) for directory in ('csrc/tpc/swa128_ilp', 'benchmarks/swa128_ilp')
            for p in sorted((root / directory).iterdir()) if p.is_file()]
hashes = {}
for name in sources:
    data = (root / name).read_bytes()
    assert data == subprocess.check_output(['git', 'show', commit + ':' + name], cwd=root), 'commit source first'
    hashes[name] = hashlib.sha256(data).hexdigest()
    target = out / 'source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
commands = []
variants = {'baseline': 'csrc/tpc/swa128_attention/head_window_fast.c'}
variants.update({p.stem: str(p.relative_to(root)) for p in sorted((root / 'csrc/tpc/swa128_ilp').glob('*.c'))})
for name, source in variants.items():
    for mode, suffix in [('-c', '.o'), ('-S', '.s')]:
        commands.append(['/usr/bin/tpc-clang', '-O2', '-ffp-contract=off', '-mcpu=gaudi2',
                         mode, str(root / source), '-o', str(out / (name + suffix))])
commands.append(['g++', '-O2', '-std=c++17', '-I/usr/lib/habanatools/include',
                 str(root / 'benchmarks/swa128_ilp/simulator.cpp'), '-L/usr/lib/habanatools',
                 '-ltpc_tests_core_ext', '-Wl,-rpath,/usr/lib/habanatools', '-o', str(out / 'simulator')])
record = {'source_commit': commit, 'sources_sha256': hashes, 'commands': commands,
          'compiler_version': subprocess.check_output(['/usr/bin/tpc-clang', '--version'], text=True),
          'device_verified': False, 'status': 'BUILDING'}
try:
    with (out / 'build.log').open('w') as log:
        for command in commands:
            log.write(json.dumps(command) + '\n')
            log.flush()
            subprocess.run(command, stdout=log, stderr=log, check=True, cwd=root)
    record['status'] = 'COMPILED'
finally:
    record['outputs_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in out.iterdir() if p.is_file()}
    (out / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
