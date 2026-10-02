#!/usr/bin/env python3
"""Build an isolated CPU-only FMA helper and retain exact provenance."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--source-identity', type=Path, help='Committed export for a source tree without .git')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'csrc/cpu/fp32_oracle.cpp'
    if args.source_identity is not None:
        exported = json.loads(args.source_identity.read_text())
        commit = exported['git_commit']
        if len(commit) != 40 or 'csrc/cpu/fp32_oracle.cpp' not in exported['files_sha256']:
            raise ValueError('invalid committed source identity')
        for relative, digest in exported['files_sha256'].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
                raise ValueError('exported source mismatch: ' + relative)
    else:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--', str(source)], text=True)
        if dirty:
            raise RuntimeError('commit CPU helper source before building')
    args.out.mkdir(parents=True, exist_ok=False)
    compiler = str(Path(shutil.which(args.cxx)).resolve())
    copied_source = args.out / source.name
    shutil.copy2(source, copied_source)
    library = args.out / 'libgaudi_fp32_oracle_cpu.so'
    command = [compiler, '-O2', '-std=c++17', '-fPIC', '-shared', '-fno-fast-math', '-ffp-contract=off',
               str(copied_source), '-o', str(library), '-lm']
    result = subprocess.run(command, capture_output=True, text=True)
    (args.out / 'build.log').write_text(result.stdout + result.stderr)
    identity = dict(source_commit=commit, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    command=command, compiler=compiler, compiler_sha256=hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
                    compiler_version=subprocess.check_output([compiler, '--version'], text=True), returncode=result.returncode,
                    scope='CPU FMA oracle only; no HPU acquisition or serving operator')
    if result.returncode == 0:
        identity['library_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
        identity['ldd'] = subprocess.check_output(['ldd', str(library)], text=True)
        runtime_paths = sorted(set(re.findall(r'(/[^\s]+)', identity['ldd'])))
        identity['runtime_libraries'] = {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in runtime_paths if Path(path).is_file()}
    (args.out / 'build.json').write_text(json.dumps(identity, indent=2) + '\n')
    result.check_returncode()
    print(json.dumps(identity, indent=2))


if __name__ == '__main__':
    main()
