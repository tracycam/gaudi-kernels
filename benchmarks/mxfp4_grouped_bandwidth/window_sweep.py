"""Screen decoder schedules on one assigned module, with rotating original bytes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
p=argparse.ArgumentParser();p.add_argument('--module',type=int,required=True)
p.add_argument('--root',type=Path,required=True);p.add_argument('--profile',action='store_true')
p.add_argument('--builds',nargs='+',default=['base','k4','k8','k16'])
p.add_argument('--experts',nargs='+',type=int,default=[8,32]);p.add_argument('--tag',default='screen');a=p.parse_args()
root=a.root.resolve();identity=json.loads((root/'source-identity.json').read_text())
for e in a.experts:
    for i,name in enumerate(a.builds):
        assert e in (8,16,32)
        build=root/name;case=f'{a.tag}-e{e}-{i}-{name}'
        cmd=['python3','tools/run_device_probe.py','--module',str(a.module),'--case',case,'--git-commit',identity['git_commit'],'--kernel-library',str(build/'libgrouped_tpc.so'),'--',
             'python3','benchmarks/mxfp4_pipeline/launch.py','--buffers','1','--compiler-sram','--audit',*(['--profile'] if a.profile else []),'--',
             str(build/'probe'),'mme',str(e),'32','512','6144','2','1',str(32//e),'random']
        print(json.dumps(dict(case=case,command=cmd)),flush=True)
        subprocess.run(cmd,cwd=root,check=True,stdout=subprocess.DEVNULL)
        print(json.dumps(dict(case=case,state='passed')),flush=True)
