"""Single-module ABBA work-amount sweep; tasks are not assumed to bind cores."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--module',type=int,required=True)
p.add_argument('--extra',action='store_true',help='Probe the within-D0 anomaly and scratch capacity')
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args()
root=a.root.resolve();identity=json.loads((root/'source-identity.json').read_text())
cases=[]
for tasks,unroll,interleave in [(n,8,False) for n in (1,6,12,24,48)]+[(24,8,True),(24,32,False)]:
    for i,reps in enumerate((128,256,256,128)):
        cases.append(dict(name=f'write-t{tasks}-u{unroll}-i{int(interleave)}-{i}',tasks=tasks,unroll=unroll,interleave=interleave,reps=reps,profile=False))
for tasks in (1,6,12,24,48):
    cases.append(dict(name=f'profile-t{tasks}',tasks=tasks,unroll=8,interleave=False,reps=256,profile=True))
if a.extra:
    cases=[]
    for tasks,rows in [(2,512),(3,512),(4,512),(24,64),(24,2048)]:
        for i,reps in enumerate((128,256,256,128)):
            cases.append(dict(name=f'extra-t{tasks}-r{rows}-{i}',tasks=tasks,rows=rows,unroll=8,interleave=False,reps=reps,profile=False))
    for tasks in (2,3,4):
        cases.append(dict(name=f'profile-t{tasks}',tasks=tasks,rows=512,unroll=8,interleave=False,reps=256,profile=True))
(root/'scaling-cases.json').write_text(json.dumps(cases,indent=2)+'\n')
for c in cases:
    build=root/f'build-u{c["unroll"]}'
    env=dict(os.environ,GK_PORT_TASKS=str(c['tasks']))
    env.pop('GK_PORT_INTERLEAVE',None)
    if c['interleave']:env['GK_PORT_INTERLEAVE']='1'
    cmd=['python3','tools/run_device_probe.py','--module',str(a.module),'--case',c['name'],'--git-commit',identity['git_commit'],'--kernel-library',str(build/'libports.so'),'--',
         'python3','benchmarks/mxfp4_pipeline/launch.py','--buffers','1','--audit',*(['--profile'] if c['profile'] else []),'--',
         sys.executable,str(root/'benchmarks/sram_ports/launch.py'),str(build/'probe'),'write',str(c.get('rows',512)),str(c['reps']),'512','2','4096','2','64','b','bf16']
    print(json.dumps(c),flush=True)
    subprocess.run(cmd,cwd=root,env=env,check=True,stdout=subprocess.DEVNULL)
    print(json.dumps(dict(case=c['name'],state='passed')),flush=True)
