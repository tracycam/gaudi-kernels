"""Export committed source identity before rsync to a checkout without .git."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
paths=subprocess.check_output(['git','ls-files','-z','csrc','python','benchmarks','tools','setup.cfg','setup.py','pyproject.toml'],cwd=root).split(b'\0')
files={}
for value in paths:
    if not value:continue
    name=value.decode();data=(root/name).read_bytes()
    committed=subprocess.check_output(['git','show',commit+':'+name],cwd=root)
    if data!=committed:raise RuntimeError('commit source edits before exporting: '+name)
    files[name]=hashlib.sha256(data).hexdigest()
a.output.write_text(json.dumps({'git_commit':commit,'files_sha256':files},indent=2)+'\n')
print(commit)
