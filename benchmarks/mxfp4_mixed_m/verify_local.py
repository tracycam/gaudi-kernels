"""Verify complete locally retained artifacts and their committed source identity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
rows=[]
for root in a.roots:
    manifest=json.loads((root/'ASSET-SHA256.json').read_text())
    actual={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file() and f.name!='ASSET-SHA256.json'}
    assert actual==set(manifest)
    for name,m in manifest.items():
        path=(root/name).resolve();assert path.is_relative_to(root.resolve())
        assert path.stat().st_size==m['bytes'] and sha(path)==m['sha256'],path
    identity=json.loads((root/'source-identity.json').read_text());commit=identity['git_commit']
    for name,h in identity['files_sha256'].items():
        assert sha(root/name)==h
        committed=subprocess.check_output(['git','show',f'{commit}:{name}'])
        assert hashlib.sha256(committed).hexdigest()==h
    cases=[]
    for case in sorted((root/'results').glob('*')):
        meta=json.loads((case/'launch.json').read_text());end=json.loads((case/'exit.json').read_text())
        assert meta['git_commit']==commit and meta['source_identity_verified'] and meta['module']==5
        assert end['runner_exit_code']==0
        assert json.loads((case/'source-identity.json').read_text())==identity
        for field in ('script_sha256','kernel_libraries_sha256','command_files_sha256'):
            for name,h in meta[field].items():assert sha(root/name)==h
        cases.append(case.name)
    rows.append(dict(root=str(root),source_commit=commit,files=len(manifest),bytes=sum(m['bytes'] for m in manifest.values()),
                     source_files=len(identity['files_sha256']),cases=cases,manifest_sha256=sha(root/'ASSET-SHA256.json')))
r=dict(status='PASS',scope='All raw assets local, SHA verified, source compared with Git; not model acceptance',
       roots=rows,total_files=sum(x['files'] for x in rows),total_bytes=sum(x['bytes'] for x in rows),total_cases=sum(len(x['cases']) for x in rows))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='roots'},indent=2))
