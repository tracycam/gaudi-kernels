"""Copy the selected executor sources/libraries into a fresh isolated fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
src = a.source_root.resolve(strict=True)
out = a.output_dir.resolve()
out.mkdir(parents=True, exist_ok=False)
names = ['executor/batch_ops.py', 'executor/native_ops.py', 'executor/precision_ops.py',
         'executor/kernels/packing.py', 'executor/torch-build/native_mxfp4_ops.so',
         'executor/tpc/libnative_tpc.so', 'torch_ops.cpp', 'glue.cpp',
         'precision_ops.cpp', 'precision_glue.cpp', 'build_production.py',
         'production-build.json', 'production-build.log']
for directory in ('ops-build', 'precision-ops-build', 'precision-tpc', 'tpc', 'legacy-native'):
    names.extend(str(p.relative_to(src)) for p in (src/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files = {}
for name in sorted(set(names)):
    source = src/name
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    target = out/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError('source changed during snapshot: ' + name)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    files[name] = digest
(out/'snapshot.json').write_text(json.dumps({'source_root': str(src), 'files_sha256': files,
    'source_modified': False, 'device_used': False}, indent=2)+'\n')
print(json.dumps({'files': len(files), 'output': str(out)}))
