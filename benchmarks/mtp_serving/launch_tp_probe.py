"""Bounded eight-rank DFlash probe, with per-module ownership and local evidence."""
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--timeout',type=int,default=600);p.add_argument('--port',type=int,default=49879);a=p.parse_args()
assert re.fullmatch('[a-z0-9][a-z0-9-]*',a.case)and 1<=a.timeout<=900 and 1024<=a.port<65535
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
from probe_process import query,stop,quarantine,active_quarantine,identity
source=json.loads((root/'source-identity.json').read_text())
for name,want in source['files_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==want,name
out=root/'results'/a.case;out.mkdir(parents=True,exist_ok=False)
(out/'source-identity.json').write_text(json.dumps(source,indent=2)+'\n')
locks=Path.home()/'gaudi-llm-experiments/.codex-kernel-locks';locks.mkdir(parents=True,exist_ok=True);held=[]
for module in range(8):
    if active_quarantine(locks/f'module-{module}.blocked.json'):raise RuntimeError('module quarantined')
    lock=(locks/f'module-{module}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(lock)
blocked=locks/'device-query.blocked.json'
if active_quarantine(blocked):raise RuntimeError('query quarantined')
command=['hl-smi','--query-aip=module_id,index,bus_id,memory.used,utilization.aip','--format=csv,noheader']
meta=dict(source_commit=source['git_commit'],reason=None,children=[],scope='TP8 DFlash component diagnostic; not model acceptance')
pre,child=query(command,out/'device-before.txt',15);meta['preflight']=pre
if pre.get('reason')or pre.get('returncode')!=0:
    if child is not None and not pre['cleanup']['reaped']:quarantine(blocked,child,'DFlash preflight unreaped')
    (out/'exit.json').write_text(json.dumps(meta,indent=2)+'\n');raise SystemExit(125)
rows={int(v.split(',')[0]):v.split(',')for v in (out/'device-before.txt').read_text().splitlines()}
if set(rows)!=set(range(8))or any(int(r[3].split()[0])>1024 or int(r[4].split()[0])for r in rows.values()):
    meta['reason']='devices occupied';(out/'exit.json').write_text(json.dumps(meta,indent=2)+'\n');raise SystemExit(125)
children=[];logs=[];start=time.monotonic()
def interrupted(sig,frame):raise SystemExit(128+sig)
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
try:
    for rank in range(8):
        env=dict(os.environ)
        for key in ('LD_PRELOAD','PYTHONPATH','HABANA_PROFILE','HABANA_PROF_CONFIG'):env.pop(key,None)
        env.update(HABANA_VISIBLE_MODULES=str(rank),HLS_MODULE_ID=str(rank),RANK=str(rank),LOCAL_RANK='0',WORLD_SIZE='8',
                   MASTER_ADDR='127.0.0.1',MASTER_PORT=str(a.port),HABANA_LOGS=str(out/f'logs-rank{rank}'),PROBE_OUT=str(out),
                   GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                   PT_HPU_LAZY_MODE='1',PT_HPU_LAZY_ACC_PAR_MODE='0',PT_HPU_ENABLE_LAZY_COLLECTIVES='true',PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS='1')
        cmd=[sys.executable,str(Path(__file__).resolve().parent/'dflash_tp_probe.py')]
        log=(out/f'rank{rank}.log').open('wb');logs.append(log)
        c=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(c)
        meta['children'].append(dict(rank=rank,module=rank,process=identity(c.pid),command=cmd))
    (out/'launch.json').write_text(json.dumps(meta,indent=2)+'\n')
    while any(c.poll()is None for c in children):
        if any(c.poll()not in (None,0)for c in children):raise RuntimeError('rank failure')
        if time.monotonic()-start>a.timeout:raise TimeoutError('bounded TP8 timeout')
        time.sleep(.1)
    if any(c.returncode for c in children):raise RuntimeError('rank failure')
except BaseException as error:meta['reason']=repr(error)
finally:
    meta['cleanup']=[];reaped=True
    for rank,c in enumerate(children):
        record=stop(c,term_seconds=5,kill_seconds=1)if c.poll()is None else dict(reaped=True,returncode=c.returncode,pid=c.pid)
        meta['cleanup'].append(record)
        if not record['reaped']:reaped=False;quarantine(locks/f'module-{rank}.blocked.json',c,'DFlash child unreaped')
    for log in logs:log.close()
    if reaped:
        post,child=query(command,out/'device-after.txt',15);meta['postflight']=post
        if child is not None and not post['cleanup']['reaped']:quarantine(blocked,child,'DFlash postflight unreaped')
    meta.update(elapsed_s=time.monotonic()-start,runner_exit_code=1 if meta['reason']or not reaped else 0)
    (out/'exit.json').write_text(json.dumps(meta,indent=2)+'\n')
raise SystemExit(meta['runner_exit_code'])
