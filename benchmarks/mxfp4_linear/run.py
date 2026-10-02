"""Sequential explicit-module coverage runs; preserve each failure independently."""
import argparse, json, os, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',default='builds/v8');p.add_argument('--commit',required=True)
p.add_argument('--suite',choices=['gates','coverage','groups','tuned','audit'],required=True);p.add_argument('--module',type=int,choices=[7],required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[2]
cases=[]
if a.suite=='gates':
    cases=[('decode-exhaustive','decode','1',4016,1,'exhaustive',512,0),
           ('tpc-tail','tpc','3',513,257,'random',256,1),
           ('mme-zero','mme','8',257,65,'zero',256,1),
           ('tpc-zero','tpc','8',257,65,'zero',256,1),
           ('mme-cancel','mme','16',513,256,'cancel',256,0),
           ('tpc-cancel','tpc','16',513,256,'cancel',256,0),
           ('mme-impulse','mme','32',513,257,'impulse',256,1)]
elif a.suite=='coverage':
    for shape,n,k in [('gateup',512,6144),('down',6144,256)]:
        for m in [1,2,8,16,32,64,128,256,257]:
            for mode in ['mme','tpc']:
                cases.append((f'{shape}-{mode}-m{m}',mode,str(m),n,k,'random',512,0))
elif a.suite=='tuned':
    for m in [1,2,8,16,32,64,128,256,257]:
        cases.append((f'down-mme-wide-m{m}','mme',str(m),6144,256,'random',6144,0))
elif a.suite=='groups':
    for shape,n,k in [('gateup',512,6144),('down',6144,256)]:
        for label,ms in [('unequal','1,0,3,17'),('sparse','0,0,32,0'),('balanced','11,10,11,10')]:
            for mode in ['mme','tpc']:
                cases.append((f'{shape}-{mode}-{label}',mode,ms,n,k,'random',512 if shape=='gateup' else 6144,1))
else:cases=[('audit-down','mme','32',6144,256,'random',6144,0),('audit-grouped','mme','1,0,3,17',513,257,'random',256,1),('audit-gateup','mme','32',512,6144,'random',512,0)]
for case,*args in cases:
    command=[sys.executable,str(root/'tools/run_device_probe.py'),'--module',str(a.module),'--case',case+'-'+Path(a.build).name,
             '--git-commit',a.commit,'--timeout','900','--kernel-library',a.build+'/libgaudi_mxfp4_linear_tpc.so','--']
    command += [sys.executable,'benchmarks/mxfp4_linear/with_audit.py'] if a.suite=='audit' else []
    command += [a.build+'/mxfp4_probe',*map(str,args)]
    print(json.dumps({'case':case,'command':command}),flush=True)
    result=subprocess.run(command,cwd=root)
    if result.returncode:raise SystemExit(result.returncode)
