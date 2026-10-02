"""Verify downloaded experiments and existing local fixture/runtime owners.

Read-only remote walk. No deletion, device acquire, or upload. Run rsync first;
the caller may retain additional local analyses beside the downloaded files.
"""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--local',type=Path,required=True);p.add_argument('--remote',required=True)
p.add_argument('--fixture-local',type=Path,required=True);p.add_argument('--fixture-remote',required=True)
p.add_argument('--runtime-freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=22);p.add_argument('--control',default='/tmp/codex-fp8feed-control');a=p.parse_args()
runtime=json.loads((a.runtime_freeze/'manifest.json').read_text())
remote_script='''import hashlib,json,sys
from pathlib import Path
def digest(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def walk(root):
 root=Path(root);out={}
 for p in sorted(root.rglob('*')):
  if not p.is_file()or '__pycache__' in p.parts:continue
  assert not p.is_symlink(),str(p)
  out[str(p.relative_to(root))]=digest(p)
 return out
config=json.loads(sys.stdin.read())
print(json.dumps(dict(experiment=walk(config['root']),fixture=walk(config['fixture']),runtime={k:digest(Path(v['source']))for k,v in config['runtime'].items()})))
'''
ssh=['ssh','-S',a.control,'-p',str(a.port),'-o','BatchMode=yes',a.host]
raw=subprocess.check_output([*ssh,'python3 -c '+shlex.quote(remote_script)],input=json.dumps(dict(root=a.remote,fixture=a.fixture_remote,runtime=runtime)).encode())
manifest=json.loads(raw)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
for kind,root in [('experiment',a.local),('fixture',a.fixture_local),('runtime',a.runtime_freeze)]:
    for name,entry in manifest[kind].items():
        path=(root/name).resolve();assert path.is_relative_to(root.resolve())
        assert digest(path)==entry,(kind,name)
        if kind=='runtime':assert entry['sha256']==runtime[name]['sha256'],name
(a.local/'remote-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
device=subprocess.check_output([*ssh,'hl-smi --query-aip=module_id,index,bus_id,memory.used,utilization.aip --format=csv,noheader'],text=True)
(a.local/'device-final.txt').write_text(device)
result=dict(status='PASS_LOCAL_ASSET_SEAL',root=str(a.local),remote=a.remote,
    remote_files=len(manifest['experiment']),remote_bytes=sum(v['bytes']for v in manifest['experiment'].values()),
    remote_local_sha256_equal=True,fixture_root=str(a.fixture_local),fixture_files=len(manifest['fixture']),fixture_sha256_equal=True,
    runtime_root=str(a.runtime_freeze),runtime_files=len(manifest['runtime']),runtime_sha256_equal=True,device_final=device,
    persistent_user_requirement='Keep source, assembly, ELF, inputs, outputs, successes, failures and Git results locally; never delete experiment assets.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
