"""Bounded comparison set; expected historical arithmetic failures stay failures."""
import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--suite',choices=['gates','speed','profile'],required=True);a=p.parse_args()
cases=[]
if a.suite=='gates':
 for mode in ['guarded','exact512','history']:
  for pattern in ['overflow','underflow']:
   cases.append((f'{mode}-{pattern}',mode,1,512,256,pattern,0,0,3 if mode=='history' else 0))
 for mode in ['guarded','exact512','history']:
  cases.append((mode+'-tail',mode,3,513,257,'random',0,1,0))
 cases += [('guarded-cancel','guarded',2,513,256,'cancel',0,0,0),('guarded-zero','guarded',2,513,257,'zero',0,1,0),('prologue-zero','prologue',1,512,6144,'zero',0,0,0)]
elif a.suite=='speed':
 for n,k in [(512,6144),(6144,256)]:
  for m in [1,2,8,32]:
   for mode in ['tpc','mme','history','guarded','exact512']:
    cases.append((f'n{n}-k{k}-m{m}-{mode}',mode,m,n,k,'random',(512 if n==512 else 6144) if mode in ['tpc','mme'] else 0,0,0))
 for mode in ['tpc','mme','history','guarded','exact512']:
  cases.append((f'stream-{mode}',mode,1,32768,6144,'random',512 if mode in ['tpc','mme'] else 0,0,0))
else:
 cases=[('profile-small-history','history',1,512,6144,'random',0,0,0),('profile-small-guarded','guarded',1,512,6144,'random',0,0,0),('profile-small-prologue','prologue',1,512,6144,'zero',0,0,0),('profile-stream-history','history',1,32768,6144,'random',0,0,0)]
for case,mode,m,n,k,pattern,tile,bias,expected in cases:
 cmd=[sys.executable,'benchmarks/mxfp4_overhead/run.py','--module','7','--build',a.build,'--case',case+'-'+a.build.split('/')[-1]]
 if a.suite=='profile':cmd.append('--profile')
 cmd+=['--',mode,str(m),str(n),str(k),pattern,str(tile),str(bias)]
 print(case,flush=True);result=subprocess.run(cmd)
 if result.returncode!=expected:raise SystemExit(result.returncode or 1)
