"""Verify frozen sources and hash every local artifact, including failures."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--assets', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = a.assets.resolve()
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
sources = []
for manifest in sorted((root/'model').glob('*/source/sha256.json')):
    records = json.loads(manifest.read_text())
    for name, expected in records.items():
        path = manifest.parent/name
        assert path.resolve().is_relative_to(manifest.parent.resolve()) and sha(path) == expected, path
    sources.append(dict(case=manifest.parent.parent.name, verified_files=len(records)))
assert len(sources) == 4
model_expected = {'moe-stream-preflight-a': (0, 'DIAGNOSTIC'), 'moe-stream-70-a': (1, 'FAIL'),
                  'moe-stream-batch-70-b': (0, 'DIAGNOSTIC_BATCH_ONLY'), 'moe-stream-audit-preflight-c': (0, 'DIAGNOSTIC')}
models = []
for case, (code, status) in model_expected.items():
    result = json.loads((root/'model'/case/'result.json').read_text())
    exit_record = json.loads((root/'model-runner'/(case+'.exit.json')).read_text())
    assert result['status'] == status and exit_record['returncode'] == code
    assert exit_record['stage'] == 'finished' and not exit_record['remaining_owned_processes']
    assert not result['candidate_accepted']
    models.append(dict(case=case, status=status, returncode=code, model_candidate_accepted=False))
for name, expected in {
    'builds/k8-a/gaudi_moe_stream_views.so': '76ef6a0c3711ac024b13d6867dea3274e30679ff8bc455bec51cb002f3b1bf9c',
    'builds/bundle-k8-a/libgaudi_moe_stream_bundle_tpc.so': '3f54fbcb981a1f1f086c7ecd07e010bb9cba1717bfe25cfdf3d3f02bf53f88dd',
}.items():
    assert sha(root/name) == expected
files = []
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name != 'local-asset-manifest.json':
        files.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path)))
manifest = root/'local-asset-manifest.json'
manifest.write_text(json.dumps(dict(files=files), indent=2)+'\n')
report = dict(status='PASS_LOCAL_ASSET_INTEGRITY', files=len(files), bytes=sum(f['bytes'] for f in files),
              manifest_sha256=sha(manifest), frozen_model_sources=sources, models=models,
              scope='All local experiment assets, failed trials retained; no model acceptance promotion')
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
