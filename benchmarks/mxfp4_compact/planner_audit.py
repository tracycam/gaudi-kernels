"""Audit actual C++ host planner partitions without Synapse runtime/device."""
import argparse,json,hashlib,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
results=[]
for n in [1,127,256,511,512,513,1024,1025]:
 for k in [1,31,32,33,64,65,257]:
  for tile in [256,512]:
   call=subprocess.run([str(a.probe.resolve()),str(n),str(k),str(tile)],capture_output=True,text=True,check=True)
   tiles=[list(map(int,row.split())) for row in call.stdout.splitlines()]
   coverage=np.zeros((n,k),np.uint8)
   for kind,nb,nc,kb,kc,block,pair in tiles:
    assert nb>=0 and nc>0 and nb+nc<=n and kb>=0 and kc>0 and kb+kc<=k
    coverage[nb:nb+nc,kb:kb+kc]+=1
    assert nc*kc*2<=16*1024*1024
    if kind==0:
     assert nc in(256,512) and kb==0 and kc%32==0
     assert block==nb//512 and pair==nb%512//256
    else: assert kc<32 or kc%32==0
   assert np.all(coverage==1)
   results.append({'n':n,'k':k,'native_n_tile':tile,'tiles':tiles,'logical_rectangle_covered_once':True})
# No huge allocations: planner rejects bad geometry and overflow before making nodes.
for n,k,tile in [(0,32,512),(1,0,512),(512,32,128),(2147483647,3,512),(512,32768,512)]:
 call=subprocess.run([str(a.probe.resolve()),str(n),str(k),str(tile)],capture_output=True,text=True)
 assert call.returncode==1
 results.append({'n':n,'k':k,'native_n_tile':tile,'expected_reject':call.stderr.strip()})
# A real useful K20480 needs N256 for explicit16MiB scratch; rejection/acceptance is deliberate.
for tile in (256,512):
 call=subprocess.run([str(a.probe.resolve()),'512','20480',str(tile)],capture_output=True,text=True)
 assert call.returncode==(0 if tile==256 else 1)
 results.append({'n':512,'k':20480,'native_n_tile':tile,'returncode':call.returncode,'stdout':call.stdout,'stderr':call.stderr})
call=subprocess.run([str(a.probe.resolve()),'513','257','512','2048'],capture_output=True,text=True,check=True)
tiles=[list(map(int,row.split())) for row in call.stdout.splitlines()]
assert len(tiles)==3 and tiles[-1][1:5]==[0,513,256,1]
results.append({'n':513,'k':257,'native_n_tile':512,'row_n_tile':2048,'tiles':tiles,'short_k_tail_coalesced':True})
a.output.write_text(json.dumps({'state':'offline_plan_pass_device_graph_unvalidated','cases':results,
 'probe_sha256':hashlib.sha256(a.probe.read_bytes()).hexdigest()},indent=2)+'\n')
print(json.dumps({'state':'offline_plan_pass_device_graph_unvalidated','cases':len(results)}))
