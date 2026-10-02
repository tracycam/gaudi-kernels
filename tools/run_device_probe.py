"""Bounded single-module probes; own-process cleanup and immutable case outputs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from probe_process import active_quarantine, quarantine, query, stop

p=argparse.ArgumentParser()
p.add_argument('--module',type=int,choices=range(8),required=True)
p.add_argument('--case',required=True)
p.add_argument('--git-commit',required=True)
p.add_argument('--timeout',type=int,default=300)
p.add_argument('--state-timeout',type=int,default=15)
p.add_argument('--kernel-library',type=Path,action='append',default=[])
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
assert re.fullmatch('[a-z0-9][a-z0-9-]*',a.case)
assert re.fullmatch('[0-9a-f]{7,40}',a.git_commit)
assert 1 <= a.timeout <= 900
assert 1 <= a.state_timeout <= 60
command=a.command[1:] if a.command[:1]==['--'] else a.command
assert command
root=Path(__file__).resolve().parents[1]
identity_path=root/'source-identity.json'
if identity_path.is_file():
    identity=json.loads(identity_path.read_text())
    assert identity['git_commit'].startswith(a.git_commit),'declared commit disagrees with exported source identity'
    for name,digest in identity['files_sha256'].items():
        source=(root/name).resolve()
        assert source.is_relative_to(root) and source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==digest, 'source changed after export: '+name
    source_verified=True
elif (root/'.git').exists():
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    assert actual.startswith(a.git_commit),'probe commit must be current Git HEAD'
    source_verified=False  # Per-file scripts/artifacts are still hashed below.
else:
    raise RuntimeError('missing source-identity.json; export committed sources before remote probes')
locks=Path.home()/'gaudi-llm-experiments/.codex-kernel-locks'
locks.mkdir(parents=True,exist_ok=True)
lock=(locks/f'module-{a.module}.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

out=root/'results'/a.case
out.mkdir(parents=True,exist_ok=False)
if identity_path.is_file():(out/'source-identity.json').write_bytes(identity_path.read_bytes())
state_quarantine=locks/'device-query.blocked.json'
module_quarantine=locks/f'module-{a.module}.blocked.json'
state_command=['hl-smi','--query-aip=module_id,index,bus_id,memory.used,utilization.aip','--format=csv,noheader']
reason=None
def interrupted(signum,frame):
    global reason
    reason=f'signal {signum}'
    raise SystemExit(128+signum)
for signum in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(signum,interrupted)
preflight={'module':a.module,'index':None,'bus_id':None,'command':command,
           'git_commit':a.git_commit,'kernel_launched':False,'returncode':None,
           'source_identity_verified':source_verified,'stage':'preflight',
           'scope':'single-device operator probe; not model acceptance'}
(out/'launch.json').write_text(json.dumps(preflight,indent=2)+'\n')
for blocked_path in [state_quarantine,module_quarantine]:
    blocked=active_quarantine(blocked_path)
    if blocked:
        preflight.update(reason='unreaped_process_quarantine',quarantine=blocked,runner_exit_code=125)
        (out/'exit.json').write_text(json.dumps(preflight,indent=2)+'\n')
        print(json.dumps(preflight),flush=True);raise SystemExit(125)
state_result,state_child=query(state_command,out/'device-before.txt',a.state_timeout)
preflight['device_query']=state_result
if state_result.get('reason') or state_result.get('returncode')!=0:
    if state_child is not None and not state_result['cleanup']['reaped']:
        preflight['quarantine']=quarantine(state_quarantine,state_child,'device query unreaped')
    preflight.update(reason='device_preflight_failed',runner_exit_code=state_result.get('interrupt_code',125))
    (out/'exit.json').write_text(json.dumps(preflight,indent=2)+'\n')
    print(json.dumps(preflight),flush=True);raise SystemExit(preflight['runner_exit_code'])
before=(out/'device-before.txt').read_text()
try:
    row=next(v.split(',') for v in before.splitlines() if int(v.split(',')[0])==a.module)
    available=int(row[3].split()[0])<=1024 and int(row[4].split()[0])==0
except (ValueError,IndexError,StopIteration):
    available=False;row=None
if not available:
    preflight.update(reason='assigned_module_occupied_or_unidentified',device_row=row,runner_exit_code=125)
    (out/'exit.json').write_text(json.dumps(preflight,indent=2)+'\n')
    print(json.dumps(preflight),flush=True);raise SystemExit(125)
env=dict(os.environ)
for key in ['LD_PRELOAD','PYTHONPATH','HABANA_PROFILE','HABANA_PROF_CONFIG']:
    env.pop(key,None)
env.update(HABANA_VISIBLE_MODULES=str(a.module),GAUDI_KERNELS_MODULE_ID=str(a.module),
    KERNEL_PROBE_OUT=str(out),PROBE_OUT=str(out),HABANA_LOGS=str(out/'habana-logs'),
    PT_HPU_LAZY_MODE='1',PT_HPU_LAZY_ACC_PAR_MODE='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
    GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so')
libraries={}
for name in a.kernel_library:
    path=(root/name).resolve();assert path.is_relative_to(root) and path.is_file()
    libraries[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    env['GC_KERNEL_PATH']+=':'+str(path)
sources={};command_files={}
for value in command:
    path=(root/value).resolve()
    if path.is_relative_to(root) and path.is_file():
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        command_files[str(path.relative_to(root))]=digest
        if path.suffix in ['.py','.sh']:sources[str(path.relative_to(root))]=digest
    if value.startswith('LD_PRELOAD='):
        for item in value.split('=',1)[1].split(':'):
            preload=(root/item).resolve()
            if preload.is_relative_to(root) and preload.is_file():
                command_files[str(preload.relative_to(root))]=hashlib.sha256(preload.read_bytes()).hexdigest()
metadata={'module':a.module,'index':int(row[1]),'bus_id':row[2].strip(),'git_commit':a.git_commit,
    'stage':'device_command','kernel_launched':False,'device_command_started':False,
    'launch_field_scope':'kernel_launched records payload start, not proof of a device launch',
    'command':command,'script_sha256':sources,'kernel_libraries_sha256':libraries,
    'command_files_sha256':command_files,
    'source_identity_verified':source_verified,'preflight':state_result,
    'scope':'single-device operator probe; not model acceptance',
    'controls':{k:v for k,v in env.items() if k.startswith(('GK_','GAUDI_KERNELS_','MAC_'))}}
(out/'launch.json').write_text(json.dumps(metadata,indent=2)+'\n')
start=time.monotonic();code=None;child=None
try:
    with (out/'run.log').open('wb') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        metadata.update(kernel_launched=True,device_command_started=True)
        (out/'launch.json').write_text(json.dumps(metadata,indent=2)+'\n')
        try:code=child.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:reason='timeout'
except OSError as exc:
    reason='device_command_start_failed'
    metadata['start_error']=str(exc)
finally:
    if child is not None and child.poll() is None:
        cleanup=stop(child,term_seconds=5,kill_seconds=1)
        metadata['cleanup']=cleanup
        if not cleanup['reaped']:
            metadata['quarantine']=quarantine(module_quarantine,child,'device probe unreaped after termination')
    if code is None and child is not None:code=child.returncode
    runner_code=124 if reason=='timeout' else (128+int(reason.split()[1]) if reason and reason.startswith('signal ') else (code if code is not None else 1))
    metadata.update(returncode=code,runner_exit_code=runner_code,reason=reason,elapsed_s=time.monotonic()-start)
    (out/'exit.json').write_text(json.dumps(metadata,indent=2)+'\n')
    # An unreaped acquisition or an already-quarantined query must not spawn
    # another potentially blocked hl-smi. Exit evidence has already been saved.
    if metadata.get('quarantine') or active_quarantine(state_quarantine):
        metadata['postflight']={'skipped':'unreaped process'}
    else:
        after,after_child=query(state_command,out/'device-after.txt',a.state_timeout)
        metadata['postflight']=after
        if after_child is not None and not after['cleanup']['reaped']:
            metadata['postflight']['quarantine']=quarantine(state_quarantine,after_child,'postflight query unreaped')
    (out/'exit.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata),flush=True)
raise SystemExit(runner_code)
