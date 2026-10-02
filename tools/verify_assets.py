"""Verify imported sources and local binary/evidence assets without using HPU."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'manifests/imports.json').read_text())
failures=[]
for entry in manifest['files']:
    path=(root/entry['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failures.append({'path':entry['path'],'error':'missing or invalid path'});continue
    if path.stat().st_size!=entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
        failures.append({'path':entry['path'],'error':'content mismatch'})
result={'status':'FAIL' if failures else 'PASS','files_checked':len(manifest['files']),
    'failures':failures,'scope':'artifact identity, not runtime or model quality'}
print(json.dumps(result,indent=2))
raise SystemExit(bool(failures))
