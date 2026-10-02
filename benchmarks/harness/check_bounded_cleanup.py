"""CPU-only regression gates for the observed preflight and SIGKILL hangs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from unittest import mock

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
import probe_process


class Unreapable:
    pid=987654321
    returncode=None
    def poll(self):return None
    def wait(self,timeout=None):
        assert timeout is not None, 'unbounded wait'
        raise subprocess.TimeoutExpired('simulated D-state child',timeout)


with mock.patch.object(probe_process.os,'killpg') as signals:
    unresponsive=probe_process.stop(Unreapable())
    assert not unresponsive['reaped'] and unresponsive['signals']==[15,9]
    assert signals.call_count==2

read_text=Path.read_text
def restricted_boot_id(path,*args,**kwargs):
    if str(path)=='/proc/sys/kernel/random/boot_id':raise PermissionError('simulated restricted sysctl')
    return read_text(path,*args,**kwargs)
with mock.patch.object(Path,'read_text',restricted_boot_id):
    own_identity=probe_process.identity(os.getpid())
    assert own_identity['pid']==os.getpid() and own_identity['boot_id'] is None

checks=[]
with tempfile.TemporaryDirectory(prefix='gaudi-probe-lifecycle-') as temporary:
    task_root=Path(temporary)
    (task_root/'tools').mkdir();(task_root/'bin').mkdir()
    for name in ['run_device_probe.py','probe_process.py']:
        shutil.copy2(root/'tools'/name,task_root/'tools'/name)
    files={str(p.relative_to(task_root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (task_root/'tools').glob('*.py')}
    (task_root/'source-identity.json').write_text(json.dumps({'git_commit':'a'*40,'files_sha256':files}))
    # Patch Path.home only inside this test interpreter. Never replace $HOME.
    (task_root/'bootstrap.py').write_text('''import runpy,sys
from pathlib import Path
base=Path(__file__).resolve().parent
Path.home=classmethod(lambda cls:base/'test-home')
sys.path.insert(0,str(base/'tools'))
sys.argv=[str(base/'tools/run_device_probe.py'),*sys.argv[1:]]
runpy.run_path(sys.argv[0],run_name='__main__')
''')
    fake=task_root/'bin/hl-smi'
    fake.write_text(f'#!{sys.executable}\n'+'''import os,signal,time
from pathlib import Path
mode=os.environ.get('FAKE_STATE','idle')
counter=Path(os.environ['FAKE_COUNTER'])
n=int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(n))
if mode=='hang' or (mode=='post-hang' and n>1):
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 time.sleep(30)
print('4, 6, 0000:cc:00.0, '+('2048' if mode=='busy' else '768')+' MiB, 0 %')
''');fake.chmod(0o755)
    (task_root/'payload.py').write_text('''import os,signal,time
from pathlib import Path
Path(os.environ['PROBE_OUT'],'payload-started').write_text('yes')
if os.environ.get('PAYLOAD_TIMEOUT')=='1':
 signal.signal(signal.SIGTERM,lambda *args:exit(0))
 time.sleep(30)
''')
    for case,state,payload_timeout,expected in [
        ('preflight-timeout','hang','0',125),('occupied','busy','0',125),
        ('timeout-zero-child','idle','1',124),('postflight-timeout','post-hang','0',0),
        ('missing-command','idle','0',1)]:
        env=dict(os.environ,PATH=str(task_root/'bin')+os.pathsep+os.environ['PATH'],
                 FAKE_STATE=state,FAKE_COUNTER=str(task_root/(case+'.count')),
                 PAYLOAD_TIMEOUT=payload_timeout)
        start=time.monotonic()
        payload=[str(task_root/'missing')] if case=='missing-command' else [sys.executable,str(task_root/'payload.py')]
        result=subprocess.run([sys.executable,str(task_root/'bootstrap.py'),'--module','4',
            '--case',case,'--git-commit','a'*40,'--timeout','1','--state-timeout','1',
            '--',*payload],env=env,text=True,capture_output=True,timeout=9)
        elapsed=time.monotonic()-start
        assert result.returncode==expected,(case,result.stdout,result.stderr)
        data=json.loads((task_root/'results'/case/'exit.json').read_text())
        launched=(task_root/'results'/case/'payload-started').exists()
        assert launched==(state in ['idle','post-hang'] and case!='missing-command')
        assert data['kernel_launched']==launched
        if case=='missing-command':assert data['reason']=='device_command_start_failed'
        if case=='timeout-zero-child':assert data['returncode']==0 and data['reason']=='timeout'
        if case=='postflight-timeout':assert data['postflight']['reason']=='timeout'
        checks.append({'case':case,'pass':True,'elapsed_s':elapsed,'runner_exit_code':result.returncode,
                       'kernel_payload_started':launched})
    guard=task_root/'quarantine.json'
    own=type('Own',(),{'pid':os.getpid()})()
    probe_process.quarantine(guard,own,'test only')
    assert probe_process.active_quarantine(guard)
    record=json.loads(guard.read_text());record['process']['start_ticks']='-1'
    guard.write_text(json.dumps(record))
    assert probe_process.active_quarantine(guard) is None and not guard.exists()
    probe_process.quarantine(guard,own,'test boot identity only')
    record=json.loads(guard.read_text());record['process']['boot_id']='previous-boot'
    guard.write_text(json.dumps(record))
    assert probe_process.active_quarantine(guard) is None and not guard.exists()
print(json.dumps({'status':'PASS','unreapable_cleanup_simulated':unresponsive,
                  'pid_reuse_guard':True,'reboot_guard':True,'restricted_boot_id_fallback':True,'checks':checks},indent=2))
