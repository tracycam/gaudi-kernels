import sys
"""Owned process group, bounded model run/query/cleanup, no device reset."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
from tools.validation.executor.probe_process import active_quarantine, identity, query, quarantine, stop
from tools.validation.executor.owned_run_state import save_state

def bounded_timeout(value):
    seconds=int(value)
    if not 60<=seconds<=7200:
        raise argparse.ArgumentTypeError('timeout must be explicitly bounded to 60..7200 seconds')
    return seconds


parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('mode',choices=['native','baseline'])
parser.add_argument('--timeout-seconds',type=bounded_timeout,default=1800,
                    help='Owned process-group wall budget, 60..7200 seconds; default remains 1800')
parser.add_argument('--output-root', type=Path, required=True)
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--plugin-source', type=Path, required=True)
parser.add_argument('--vllm-source', type=Path, required=True)
parser.add_argument('--benchmark-module', default='tools.validation.executor.native_service_test',
                    choices=('tools.validation.executor.bridge_collective_test',
                             'tools.validation.executor.bridge_model_test',
                             'tools.validation.executor.native_service_test',
                             'tools.validation.executor.migration_latency_repeat',
                             'tools.validation.executor.packed_model_probe',
                             'tools.validation.executor.native_batch_test',
                             'tools.validation.executor.prefill_dispatch_probe'))
args,extra=parser.parse_known_args();root=args.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
source_root=Path(__file__).resolve().parents[3]
source_identity=source_root/'source-identity.json'
source_commit=os.getenv('SOURCE_GIT_COMMIT')
if source_identity.is_file():
    exported=json.loads(source_identity.read_text())['git_commit']
    if source_commit and not exported.startswith(source_commit):
        raise ValueError('Declared source commit does not match exported source')
    source_commit=exported

assert '/' not in args.case and not (root/args.case).exists()
lock_root=Path.home()/'gaudi-llm-experiments/.codex-kernel-locks'
lock_root.mkdir(parents=True,exist_ok=True)
locks=[]
for module in range(8):
    if active_quarantine(lock_root/f'module-{module}.blocked.json'):
        raise RuntimeError('module quarantined: '+str(module))
    handle=(lock_root/f'module-{module}.lock').open('a+')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
blocked=lock_root/'device-query.blocked.json'
if active_quarantine(blocked):raise RuntimeError('device query quarantined')
command=['hl-smi','--query-aip=index,memory.used,utilization.aip','--format=csv,noheader']
start=time.monotonic();meta={'returncode':None,'reason':None,'case':args.case,
    'mode':args.mode,'arguments':extra,'source_commit':source_commit,
    'requested_modules':list(range(8)), 'timeout_seconds':args.timeout_seconds,
    'runner_process':identity(os.getpid()), 'stage':'preflight'}
def save():
    meta['elapsed_s']=time.monotonic()-start
    save_state(root/(args.case+'.exit.json'),meta)
save()
pre,probe=query(command,root/(args.case+'.device-before.txt'),15);meta['preflight']=pre
if pre.get('returncode')!=0 or pre.get('reason'):
    if probe is not None and not pre['cleanup']['reaped']:quarantine(blocked,probe,'model preflight query unreaped')
    meta.update(returncode=125,reason='preflight query failed');save();raise SystemExit(125)
rows=(root/(args.case+'.device-before.txt')).read_text().strip().splitlines()
if len(rows)!=8 or any(int(r.split(',')[1].split()[0])>1024 or int(r.split(',')[2].split()[0]) for r in rows):
    meta.update(returncode=125,reason='devices not idle');save();raise SystemExit(125)

def group_members(group):
    members=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            value=(path/'stat').read_text();fields=value[value.rfind(')')+2:].split()
            # Zombies have already closed device handles; their new parent
            # owns wait/reaping. Never treat them as live accelerator owners.
            if int(fields[2])==group and fields[0]!='Z':
                item=identity(int(path.name))
                if item:members.append(item)
        except (OSError,ValueError,IndexError):pass
    return members

child=None
def interrupted(sig,frame):
    meta['reason']='signal '+str(sig)
    raise SystemExit(128+sig)
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
try:
    with (root/(args.case+'.log')).open('wb') as out:
        source_root=Path(__file__).resolve().parents[3]
        child_environment=dict(os.environ)
        child_environment['PYTHONPATH']=str(source_root/'python')+os.pathsep+str(source_root)

        child=subprocess.Popen([sys.executable, '-m', 'tools.validation.launch',
                                '--manifest', str(args.manifest.resolve(strict=True)),
                                '--plugin-source', str(args.plugin_source.resolve(strict=True)),
                                '--vllm-source', str(args.vllm_source.resolve(strict=True)),
                                '--benchmark-module', args.benchmark_module,
                                '--', *(['--native'] if args.mode=='native' and args.benchmark_module=='tools.validation.executor.native_service_test' else []), *extra],cwd=root,
                               stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=child_environment)
        meta['process']=identity(child.pid);meta['stage']='running';save();offset=0
        while child.poll() is None:
            time.sleep(1)
            with (root/(args.case+'.log')).open('rb') as src:src.seek(offset);chunk=src.read();offset=src.tell()
            fatal=any(x in chunk for x in [b'ValidateSyncInputTensors tensor_data is empty',b'Engine core initialization failed.',b'Synapse detected a device critical error',b'PT_DEVMEM OOM',b'FIXED_PLAN_REPLAY_ERROR'])
            fatal=fatal or b'RuntimeError: Empty tensor optional' in chunk
            fatal=fatal or b'is not serializableSet VLLM_ALLOW_INSECURE_SERIALIZATION' in chunk
            fatal=fatal or (b'E1_GATE' in chunk and b'"status": "FAIL"' in chunk)
            if fatal or time.monotonic()-start>args.timeout_seconds:
                meta['reason']='fatal rank error' if fatal else str(args.timeout_seconds)+' second timeout'
                break
except BaseException as error:
    meta['reason']=meta['reason'] or repr(error)
finally:
    if child is not None:
        meta['cleanup']=stop(child,term_seconds=5,kill_seconds=5) if child.poll() is None else {'pid':child.pid,'reaped':True,'returncode':child.returncode}
        if child.returncode or meta['reason']:
            # Only our own process group. Keep identities of surviving workers;
            # killing the launcher alone is not proof that devices are released.
            for sig in (signal.SIGTERM,signal.SIGKILL):
                members=group_members(child.pid)
                if not members:break
                try:os.killpg(child.pid,sig)
                except ProcessLookupError:pass
                deadline=time.monotonic()+5
                while group_members(child.pid) and time.monotonic()<deadline:time.sleep(.1)
        if not child.returncode and not meta['reason']:
            deadline=time.monotonic()+5
            while group_members(child.pid) and time.monotonic()<deadline:time.sleep(.1)
        remaining=group_members(child.pid);meta['remaining_owned_processes']=remaining
        if remaining:
            record={'process':remaining[0],'processes':remaining,'reason':'model process group not fully reaped'}
            for module in range(8):
                path=lock_root/f'module-{module}.blocked.json';temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path)
            meta['reason']=meta['reason'] or 'owned workers remain'
        meta['returncode']=child.returncode if child.returncode is not None else 124
        if not remaining and meta['cleanup']['reaped']:
            post,probe=query(command,root/(args.case+'.device-after.txt'),15);meta['postflight']=post
            if post.get('returncode')!=0 or post.get('reason'):
                meta['reason']=meta['reason'] or 'postflight query failed'
                if probe is not None and not post['cleanup']['reaped']:quarantine(blocked,probe,'model postflight query unreaped')
    if meta['reason'] and meta['returncode']==0:meta['returncode']=1
    meta['stage']='finished'
    save()
raise SystemExit(meta['returncode'] if meta['returncode'] is not None else 1)
