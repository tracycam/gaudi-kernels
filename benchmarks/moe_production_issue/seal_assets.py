"""Copy self-produced local/remote evidence to a fresh canonical archive and hash it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
p=argparse.ArgumentParser();p.add_argument('--build-root',type=Path,required=True);p.add_argument('--canonical-repo',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
relative=Path('artifacts/builds/moe-production-issue/final-a');dest=a.canonical_repo/relative
dest.mkdir(parents=True,exist_ok=False)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()
remote=json.loads((a.build_root/'remote-parent-manifest.json').read_text())
for entry in remote['files']:
    path=a.build_root/'remote-all'/entry['path']
    assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'],path
shutil.copytree(a.build_root/'remote-all',dest/'remote')
(dest/'offline').mkdir()
for source in sorted(a.build_root.iterdir()):
    # Exact full remote targets supersede these byte-duplicate staging/retrieval trees.
    if source.name in ('remote-all','remote-final') or source.name.startswith('target-'):continue
    target=dest/'offline'/source.name
    if source.is_dir():shutil.copytree(source,target)
    else:shutil.copy2(source,target)
for entry in remote['files']:
    assert digest(dest/'remote'/entry['path'])==entry['sha256']
files=[]
for path in sorted(dest.rglob('*')):
    if path.is_file():
        local=path.relative_to(dest)
        original=a.build_root/('remote-all' if local.parts[0]=='remote' else '.')/Path(*local.parts[1:])
        sha=digest(path);assert sha==digest(original),original
        files.append(dict(path=str(path.relative_to(a.canonical_repo)),bytes=path.stat().st_size,sha256=sha))
manifest=dict(archive_root=str(relative),remote_source=remote['root'],remote_verified_files=len(remote['files']),remote_verified_bytes=remote['total_bytes'],all_canonical_files_match_local_originals=True,file_count=len(files),total_bytes=sum(x['bytes'] for x in files),files=files,scope='All remote failed/successful targets; local builds/SIM/assembly/provenance; no global model qualification')
a.manifest.parent.mkdir(parents=True,exist_ok=True);a.manifest.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='files'}))
