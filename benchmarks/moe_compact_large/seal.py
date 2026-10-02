"""Read-only copy and SHA seal of completed probe directories and their inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser()
for key in ('host','control','remote'):p.add_argument('--'+key,required=True)
p.add_argument('--port',type=int,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--cached-fixtures',type=Path,required=True)
p.add_argument('--transfer-timeout',type=int,default=600,
               help='read-only rsync deadline; larger full-tensor archives may need longer')
a=p.parse_args()
if not 60<=a.transfer_timeout<=3600:p.error('transfer timeout must be between60 and3600 seconds')
ssh=['ssh','-S',a.control,'-o','BatchMode=yes','-o','ConnectTimeout=8','-p',str(a.port)]
script='''from pathlib import Path
import hashlib,json
r=Path(REMOTE)
cases=list((r/'results').iterdir());assert cases
for case in cases:
 if not case.is_dir():continue
 x=json.loads((case/'exit.json').read_text())
 assert x['returncode'] is not None and x.get('postflight',{}).get('returncode')==0,case
files={}
for path in sorted(r.rglob('*')):
 if not path.is_file() or '__pycache__' in path.parts:continue
 with path.open('rb')as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 files[str(path.relative_to(r))]={'bytes':path.stat().st_size,'sha256':h}
print(json.dumps(files))
'''.replace('REMOTE',repr(a.remote))
manifest=json.loads(subprocess.check_output(ssh+[a.host,'python3 -'],input=script.encode(),timeout=120))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()
a.out.mkdir(parents=True,exist_ok=True)
# Existing immutable fixture bytes are verified before seeding local hardlinks.
# rsync does not use --inplace and cannot modify those previous sealed owners.
for name,record in manifest.items():
    if not name.startswith('fixtures/'):continue
    old=a.cached_fixtures/name.removeprefix('fixtures/');dest=a.out/name
    if dest.exists() or not old.is_file():continue
    if old.stat().st_size!=record['bytes'] or digest(old)!=record['sha256']:continue
    dest.parent.mkdir(parents=True,exist_ok=True);os.link(old,dest)
subprocess.run(['rsync','-az','--checksum','--protect-args','--exclude=__pycache__',
    '-e',shlex.join(ssh),a.host+':'+a.remote+'/',str(a.out)+'/'],check=True,timeout=a.transfer_timeout)
for name,record in manifest.items():
    path=a.out/name
    assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256'],name
identity=json.loads((a.out/'source-identity.json').read_text())
for name,h in identity['files_sha256'].items():assert digest(a.out/name)==h,name
record={'status':'REMOTE_LOCAL_SHA_VERIFIED','remote':a.remote,'local':str(a.out),
    'source_commit':identity['git_commit'],'files':manifest,'file_count':len(manifest),
    'bytes':sum(x['bytes']for x in manifest.values()),'excluded':'derived __pycache__ only',
    'scope':'source, original-width weights, DLLs, full tensor inputs/outputs, graphs/profiles and successful/failed run evidence'}
(a.out/'remote-sha256.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items()if k!='files'}))
