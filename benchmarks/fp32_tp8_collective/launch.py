"""One bounded TP8 run; call only after root grants all eight modules.

Owns all standard module locks, one bounded preflight, and bounded process-group
cleanup. No resets, global process signals, unbounded hl-smi or wait().
"""
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--packet-only',action='store_true');p.add_argument('--prequeue',action='store_true');p.add_argument('--comparison',choices=['ag_sum','ar_handoff','ar_ag_consumers','ag_inplace','ag_two_stream','native_graph'],default='ag_sum');p.add_argument('--plan-only',action='store_true');p.add_argument('--pin-cpus',action='store_true');p.add_argument('--binary',type=Path);p.add_argument('--engine',choices=['native','framework'],default='native');p.add_argument('--torch-library',type=Path);p.add_argument('--api-library',type=Path,required=True);p.add_argument('--kernel-library',type=Path,required=True);p.add_argument('--extra-kernel-library',type=Path,action='append',default=[]);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--timeout',type=int,default=180);p.add_argument('--teardown-grace',type=int,default=30);p.add_argument('--repeats',type=int,default=10);p.add_argument('--trials',type=int,default=5);p.add_argument('--port',type=int,default=49873);a=p.parse_args()
assert (a.engine=='native' and a.binary) or (a.engine=='framework' and a.torch_library)
assert (a.comparison=='ag_sum' and not a.prequeue) or a.engine=='native'
assert not a.packet_only or (a.comparison in ('ag_inplace','ag_two_stream') and a.engine=='native' and not a.prequeue)
assert a.comparison!='ag_two_stream' or a.repeats<=4
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'));from probe_process import query,stop,quarantine,active_quarantine,identity
assert re.fullmatch('[a-z0-9][a-z0-9-]*',a.case) and 1<=a.timeout<=900 and 5<=a.teardown_grace<=60 and 1<=a.repeats<=100 and 1<=a.trials<=20 and 1024<=a.port<=65532
if a.plan_only:
 print(json.dumps(dict(engine=a.engine,modules=list(range(8)),hccl_dtype='FP32',hccl_count=6144,send_bytes=24576,allgather_receive_bytes=196608,stages=69,prequeue=a.prequeue,comparison=a.comparison,order=(['TPC+AG(serial)+sum','TPC+AG(explicit streams)+sum','TPC+AG(explicit streams)+sum','TPC+AG(serial)+sum'] if a.comparison=='ag_two_stream' else ['TPC+AG(out)+sum','TPC+AG(in)+sum','TPC+AG(in)+sum','TPC+AG(out)+sum'] if a.comparison=='ag_inplace' else ['TPC+AR+copy','TPC+AG+sum','TPC+AG+sum','TPC+AR+copy'] if a.comparison=='ar_ag_consumers' else ['AR', 'TPC+AR+TPC' if a.comparison=='ar_handoff' else 'AG+sum', 'TPC+AR+TPC' if a.comparison=='ar_handoff' else 'AG+sum', 'AR']),fixtures=['exact','ordinary'],timeout=a.timeout,device_contacted=False),indent=2));raise SystemExit(0)
out=root/'results'/a.case;out.mkdir(parents=True,exist_ok=False)
source=json.loads((root/'source-identity.json').read_text())
for name,want in source['files_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==want,name
(out/'source-identity.json').write_text(json.dumps(source,indent=2)+'\n')
locks=Path.home()/'gaudi-llm-experiments/.codex-kernel-locks';locks.mkdir(parents=True,exist_ok=True);held=[]
for m in range(8):
 if active_quarantine(locks/f'module-{m}.blocked.json'):raise RuntimeError('module quarantined '+str(m))
 lock=(locks/f'module-{m}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(lock)
blocked=locks/'device-query.blocked.json'
if active_quarantine(blocked):raise RuntimeError('device query quarantined')
command=['hl-smi','--query-aip=module_id,index,bus_id,memory.used,utilization.aip','--format=csv,noheader']
metadata=dict(fixture_sha256={str(f.relative_to(a.fixtures)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.fixtures.rglob('*') if f.is_file()},source_commit=source['git_commit'],packet_only=a.packet_only,prequeue=a.prequeue,comparison=a.comparison,scope=('FP32 TP8 public native graph: bare AG / explicit producer-AG-consumer / owned graph, six paired phases; no model TPS' if a.comparison=='native_graph' else 'FP32 TP8 69 same-size collectives, optional TPC producer/consumer; no model TPS'),extra_kernel_library_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in a.extra_kernel_library},children=[],engine=a.engine,binary_sha256=hashlib.sha256((a.binary if a.engine=='native' else a.torch_library).read_bytes()).hexdigest(),api_library_sha256=hashlib.sha256(a.api_library.read_bytes()).hexdigest(),kernel_library_sha256=hashlib.sha256(a.kernel_library.read_bytes()).hexdigest(),reason=None)
pre,child=query(command,out/'device-before.txt',15);metadata['preflight']=pre
if pre.get('reason') or pre.get('returncode')!=0:
 if child is not None and not pre['cleanup']['reaped']:quarantine(blocked,child,'TP8 preflight query unreaped')
 (out/'exit.json').write_text(json.dumps(metadata,indent=2)+'\n');raise SystemExit(125)
rows={int(v.split(',')[0]):v.split(',') for v in (out/'device-before.txt').read_text().splitlines()};assert set(rows)==set(range(8)) and all(int(r[3].split()[0])<=1024 and int(r[4].split()[0])==0 for r in rows.values())
children=[];logs=[];start=time.monotonic();reaped=True;available_cpus=sorted(os.sched_getaffinity(0));metadata['allowed_cpus']=available_cpus;metadata['pin_cpus']=a.pin_cpus;metadata['teardown_grace_s']=a.teardown_grace;metadata['D_state_observations']=[]
if a.pin_cpus:assert len(available_cpus)>=8
def interrupted(sig,frame):metadata['reason']='signal '+str(sig);raise SystemExit(128+sig)
for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,interrupted)
try:
 for fixture_index,kind in enumerate(['exact','ordinary']):
  directory=out/kind;directory.mkdir();(directory/'shared').write_bytes(bytes(65536));children=[]
  for rank in range(8):
   env=dict(os.environ)
   for k in ['LD_PRELOAD','HABANA_PROFILE','HABANA_PROF_CONFIG']:env.pop(k,None)
   env.update(HABANA_VISIBLE_MODULES=str(rank),HLS_MODULE_ID=str(rank),GAUDI_KERNELS_MODULE_ID=str(rank),HCCL_COMM_ID=f'127.0.0.1:{a.port+2*fixture_index}',GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so:'+str(a.kernel_library.resolve())+''.join(':'+str(p.resolve())for p in a.extra_kernel_library),HABANA_LOGS=str(directory/f'logs-rank{rank}'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
   env['TP8_PROBE_PACKET_ONLY']='1' if a.packet_only else '0'
   if a.packet_only:env.update(LOG_LEVEL_HCL='0',LOG_LEVEL_HCL_ECR='0',LOG_LEVEL_HCL_SUBMIT='0')
   env['TP8_PROBE_PREQUEUE']='1' if a.prequeue else '0';env['TP8_PROBE_COMPARISON']=a.comparison;env['LD_PRELOAD']=str(a.api_library.resolve());env['OPENBLAS_NUM_THREADS']='1'
   if a.pin_cpus:env['TP8_PROBE_CPU']=str(available_cpus[rank*len(available_cpus)//8])
   else:env.pop('TP8_PROBE_CPU',None)
   if a.engine=='native':cmd=[str(a.binary.resolve()),str(directory),str((a.fixtures/kind).resolve()),str(rank),str(a.repeats),str(a.trials)]
   else:
    env.update(RANK=str(rank),LOCAL_RANK='0',WORLD_SIZE='8',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(a.port+2*fixture_index+1),PT_HPU_LAZY_MODE='1',PT_HPU_LAZY_ACC_PAR_MODE='0',PT_HPU_ENABLE_LAZY_COLLECTIVES='true',PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS='1',PT_HPU_ACC_THREAD_VERSION='1')
    cmd=[sys.executable,str(Path(__file__).resolve().parent/'framework.py'),'--out',str(directory),'--fixture',str((a.fixtures/kind).resolve()),'--library',str(a.torch_library.resolve()),'--repeats',str(a.repeats),'--trials',str(a.trials)]
   log=(directory/f'rank{rank}.log').open('wb');logs.append(log);proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=directory);children.append(proc)
   metadata['children'].append(dict(kind=kind,rank=rank,module=rank,command=cmd,environment={k:v for k,v in env.items() if k.startswith(('LOG_LEVEL_','HCCL_','HCL_','PT_','GC_','HABANA_','HLS_','TP8_','OMP_','MKL_'))},process=identity(proc.pid)))
  uninterruptible={}
  while any(c.poll() is None for c in children):
   for rank,c in enumerate(children):
    try:
     text=Path(f'/proc/{c.pid}/stat').read_text();state=text[text.rfind(')')+2:].split()[0]
    except OSError:state=None
    if state=='D':
     first=uninterruptible.setdefault(c.pid,time.monotonic())
     try:phase=json.loads((directory/f'rank{rank}-lifecycle.json').read_text())['stage']
     except (OSError,ValueError,KeyError):phase='unknown'
     grace=a.teardown_grace if phase in ('device_release','synapse_destroy') else 5
     age=time.monotonic()-first
     if age>5 and not any(r['pid']==c.pid for r in metadata['D_state_observations']):
      try:wchan=Path(f'/proc/{c.pid}/wchan').read_text().strip()
      except OSError:wchan=None
      metadata['D_state_observations'].append(dict(pid=c.pid,phase=phase,age_s=age,grace_s=grace,wchan=wchan));(out/'D-state-observations.json').write_text(json.dumps(metadata['D_state_observations'],indent=2)+'\n')
     if age>grace:raise RuntimeError('persistent D-state pid '+str(c.pid)+' phase '+phase)
    else:uninterruptible.pop(c.pid,None)
   if any(c.poll() not in (None,0) for c in children):raise RuntimeError('rank failure')
   if time.monotonic()-start>a.timeout:metadata['reason']='timeout';raise TimeoutError('bounded TP8 timeout')
   time.sleep(.1)
  if any(c.returncode for c in children):raise RuntimeError('rank failure')
except BaseException as e:
 metadata['reason']=metadata['reason'] or repr(e)
finally:
 cleanup=[]
 for rank,c in enumerate(children):
  record=stop(c) if c.poll() is None else dict(reaped=True,returncode=c.returncode,pid=c.pid);cleanup.append(record)
  if not record['reaped']:reaped=False;quarantine(locks/f'module-{rank}.blocked.json',c,'TP8 child unreaped')
 metadata.update(cleanup=cleanup,elapsed_s=time.monotonic()-start)
 for log in logs:log.close()
 if reaped:
  post,child=query(command,out/'device-after.txt',15);metadata['postflight']=post
  if post.get('reason') or post.get('returncode')!=0:
   metadata['reason']=metadata['reason'] or 'postflight failed'
   if child is not None and not post['cleanup']['reaped']:quarantine(blocked,child,'TP8 postflight query unreaped')
 metadata['runner_exit_code']=124 if metadata['reason']=='timeout' else 1 if metadata['reason'] or not reaped else 0
 (out/'exit.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata),flush=True)
raise SystemExit(metadata['runner_exit_code'])
