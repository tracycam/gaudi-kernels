#!/usr/bin/env python3
"""Seal locally preserved successful/failed assets without deleting anything."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def seal(root):
    manifest = root / 'asset-manifest.json'
    files, links = [], []
    for path in sorted(root.rglob('*')):
        if path == manifest:
            continue
        name = str(path.relative_to(root))
        if path.is_symlink():
            links.append({'path': name, 'target': str(path.readlink())})
        elif path.is_file():
            files.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
    result = {'files': files, 'symlinks': links, 'file_count': len(files),
        'total_bytes': sum(f['bytes'] for f in files),
        'scope': 'All regular local assets; manifest itself excluded; symlink targets recorded without following'}
    manifest.write_text(json.dumps(result, indent=2) + '\n')
    return {k: v for k, v in result.items() if k not in ('files', 'symlinks')} | {'manifest_sha256': digest(manifest)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    a = p.parse_args()
    print(json.dumps(seal(a.root.resolve()), indent=2))
