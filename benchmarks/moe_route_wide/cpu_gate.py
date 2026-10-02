"""Bounded CPU route/capacity specification; no HPU import or acquisition."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'python'))
from gaudi_kernels.route_wide_metadata import geometry
from gaudi_kernels.route_metadata import geometry as original_geometry
spec=importlib.util.spec_from_file_location('wide_planner',root/'benchmarks/moe_route_tiles/planner.py');planner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=planner;spec.loader.exec_module(planner)
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path);a=p.parse_args();records=[]
def check(ids,c,kind):
 t,r=len(ids),len(ids[0]);e=384;cc,b=geometry(t,r,e,c);cap=planner.Capacity(t,r,e,c);assert cc==c and b==cap.max_tiles
 assigned=planner.assign(ids,cap);planner.validate(ids,assigned)
 active=sum(v>0 for v in assigned.tile_valid_rows);cost=cap.costs(active,len(set(v for row in ids for v in row)),n_tile=2048)
 assert max(assigned.inverse)<b*c and min(assigned.inverse)>=0
 assert sum(assigned.tile_valid_rows)==t*r
 records.append(dict(T=t,R=r,C=c,distribution=kind,**{k:cost[k]for k in('max_tiles','static_row_slots','valid_routes','actual_tiles','empty_tiles','actual_tile_padding_rows','mme_nodes','mme_node_count_kind','all_decoded_store_bytes_including_empty','active_decoded_store_bytes','empty_zero_store_bytes','gp_decoded_fragment_bytes','down_decoded_fragment_bytes','all_functional_intermediates_payload_bytes')}))
for t in (64,65,128,129,512,513):
 for c in (64,128):
  if t<c:continue
  for r in (1,2,8):
   for kind in ('uniform','hot','skew'):
    ids=planner.fixture(t,r,384,kind);check(ids,c,kind)
    check([list(reversed(row))for row in reversed(ids)],c,kind+'-changed')
rejections=0
for args in ((513,8,384,32),(513,8,384,129),(63,8,384,64),(513,8,384,64,0),(513,8,384,64,4105),(513,8,384,True),(514,8,384,64),(513,9,384,64),(513,8,7,64)):
 try:geometry(*args)
 except ValueError:rejections+=1
 else:raise AssertionError(('invalid geometry accepted',args))
for c in (64,128):
 try:original_geometry(513,8,384,c)
 except ValueError:rejections+=1
 else:raise AssertionError('default C<=32 guard changed')
checkpoint=None
if a.checkpoint:
 import torch
 digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();assert digest=='6ebc2f3734fce93bf05c05fd46f8b94997f600653ab1f6860710349aa014dcca'
 cpu=torch.load(a.checkpoint,map_location='cpu',weights_only=False);ids=cpu['ids'].tolist();assert len(ids)==512 and len(ids[0])==8
 checkpoint=dict(sha256=digest,rows=[])
 for c in (32,64,128):
  cap=planner.Capacity(512,8,384,c);ass=planner.assign(ids,cap);active=sum(v>0 for v in ass.tile_valid_rows)
  checkpoint['rows'].append(cap.costs(active,len(set(v for row in ids for v in row)),n_tile=2048))
report=dict(status='PASS_CPU_WIDE_ROUTE_SPEC',device_qualified=False,records=records,rejected_geometry=rejections,checkpoint=checkpoint,scope='Integer route coverage/order and static payload estimates; MME counts are construction estimates, not physical nodes, launches or measured time.')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],cases=len(records),rejections=rejections,checkpoint=bool(checkpoint))))
