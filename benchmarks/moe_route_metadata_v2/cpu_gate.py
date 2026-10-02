"""Staged metadata against the previously bounded independent route planner."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'benchmarks/moe_route_tiles'))
sys.path.insert(0,str(root/'python'))
from planner import Capacity,assign,fixture
from reference import reference
from gaudi_kernels.route_metadata import geometry

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();records=[];words=0
for t in (1,2,8,32,128,512,513):
 for r in (1,2,8):
  for requested in (16,32):
   cap=Capacity(t,r,384,requested);c,b=geometry(t,r,384,requested)
   assert (c,b)==(cap.rows,cap.max_tiles)
   for state in ('uniform','hot','skew'):
    ids=fixture(t,r,384,state);plan=assign(ids,cap);got=reference(ids,384,c,b)
    for name,want in [('counts',plan.counts),('tile_expert',plan.tile_expert),('tile_base',plan.tile_ordinal_base),('valid_rows',plan.tile_valid_rows),('row_map',plan.row_to_route),('inverse',plan.inverse)]:
     assert got[name]==want,(t,r,requested,state,name);words+=len(want)
    assert got['status']==[0] and got['row_status']==[0]*t
    assert [sum(x)for x in got['chunk_counts']]==got['counts'];words+=sum(len(x)for x in got['chunk_counts'])
    # Verify all fixed row-map cells have one owner, including unused tiles.
    ownership=[0]*(b*c)
    for lo,hi in zip(got['prefix'],got['prefix'][1:]):
     for i in range(lo*c,hi*c):ownership[i]+=1
    for i in range(got['prefix'][-1]*c,b*c):ownership[i]+=1
    assert ownership==[1]*(b*c)
    records.append(dict(T=t,R=r,C=c,B=b,state=state,status=0))
bad=[]
for name,ids,b,bit in [('negative',[[-1,1],[2,3]],4,1),('large',[[4,1],[2,3]],4,1),('duplicate',[[1,1],[2,3]],4,2),('overflow',[[0,1],[2,3]],1,4)]:
 got=reference(ids,4,2,b);assert got['status'][0]&bit
 assert got['row_map']==[-1]*(2*b) and got['inverse']==[-1]*4
 assert got['tile_expert']==[-1]*b and not any(got['valid_rows']) and not any(got['prefix'])
 bad.append(dict(case=name,status=got['status'][0]))
negative=0
for args in [(0,8,384,16),(514,8,384,16),(2,0,384,16),(2,9,384,16),(2,8,7,16),(2,8,385,16),(2,8,384,0),(2,8,384,33),(True,8,384,16),(2,8,384,16,0),(2,8,384,16,17)]:
 try:geometry(*args)
 except ValueError:negative+=1
 else:raise AssertionError(args)
report=dict(status='PASS_CPU_V2_PLANNER_PARITY',records=records,integer_words_checked=words,bad_inputs=bad,shape_rejections=negative,wall_seconds=time.perf_counter()-start,device_qualified=False,scope='Full fixed-output integer maps, stable route slots and write ownership; not device scheduling/performance.')
(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ('status','integer_words_checked','shape_rejections','wall_seconds')}))
