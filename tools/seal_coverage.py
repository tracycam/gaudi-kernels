"""Seal committed source history and all local experiment assets, then verify tar bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--title',default='Kernel experiment assets');a=p.parse_args()
root=Path(__file__).resolve().parents[1];destination=a.output.resolve()
if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
    raise RuntimeError('commit source/evidence changes before sealing')
destination.mkdir(parents=True,exist_ok=False)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
subprocess.run(['git','bundle','create',str(destination/'source-history.bundle'),'--all'],cwd=root,check=True)
subprocess.run(['git','archive','--format=tar','--output='+str(destination/'source.tar'),commit],cwd=root,check=True)


def digest(stream):
    sha=hashlib.sha256()
    for chunk in iter(lambda:stream.read(4*1024*1024),b''):sha.update(chunk)
    return sha.hexdigest()


manifest={}
for directory in [root/'artifacts',root/'evidence']:
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or 'archive' in path.relative_to(root).parts:continue
        if path.is_symlink():raise RuntimeError('unexpected symlink: '+str(path))
        with path.open('rb') as stream:sha=digest(stream)
        manifest[str(path.relative_to(root))]={'bytes':path.stat().st_size,'sha256':sha}
with tarfile.open(destination/'assets.tar','w') as archive:
    for name in manifest:archive.add(root/name,arcname=name,recursive=False)
checked=set()
with tarfile.open(destination/'assets.tar','r|') as archive:
    for member in archive:
        assert member.isfile() and member.name in manifest and member.name not in checked
        with archive.extractfile(member) as stream:sha=digest(stream)
        assert member.size==manifest[member.name]['bytes'] and sha==manifest[member.name]['sha256'],member.name
        checked.add(member.name)
assert checked==set(manifest)
record={'status':'VERIFIED','source_commit':commit,'files':manifest,'file_count':len(manifest),
        'uncompressed_payload_bytes':sum(v['bytes'] for v in manifest.values()),
        'scope':'source, binaries, assembly, inputs, full references/outputs and success/failure evidence; not whole-model acceptance',
        'excluded':'derived __pycache__ and redundant nested archive directories; originals remain local'}
(destination/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
outer={}
for path in sorted(destination.iterdir()):
    if path.is_file():
        with path.open('rb') as stream:outer[path.name]=digest(stream)
(destination/'SHA256SUMS.json').write_text(json.dumps(outer,indent=2)+'\n')
(destination/'README.md').write_text(f'''# {a.title}

VERIFIED: {len(manifest)} local artifact files, {record['uncompressed_payload_bytes']} bytes.
Source commit: `{commit}`. All Git branches/history are in `source-history.bundle`;
the committed working tree is in `source.tar`; original asset paths are in `assets.tar`.
Every archived asset was read back and checked against `manifest.json`.

用户持续要求：实验源码、汇编、二进制、原始输入/输出、成功/失败证据和 Git 成果留在本地。
本封存遵守该要求，不覆盖此前任何封存包。完整模型/TPS与设备动态路由尚未验收。
正式入口：../../gaudi-kernels/README.md。三线报告及框架报告在源码包 docs/ 中。
''')
print(json.dumps({k:v for k,v in record.items() if k!='files'},indent=2))
