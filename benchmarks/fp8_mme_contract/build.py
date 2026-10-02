"""Compile the native format contract probe; this never acquires a device."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
command = ['g++', '-O3', '-std=c++17', '-I/usr/include/habanalabs',
           str(root/'benchmarks/fp8_mme_contract/codes.cpp'),
           '-L/usr/lib/habanalabs', '-Wl,-rpath,/usr/lib/habanalabs',
           '-lSynapse', '-o', str(out/'codes')]
record = dict(command=command, device_acquired=False, state='building')
try:
    with (out/'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    record.update(state='built', executable_sha256=hashlib.sha256((out/'codes').read_bytes()).hexdigest())
finally:
    (out/'build.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
