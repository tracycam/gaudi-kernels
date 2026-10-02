"""Offline dispatcher/Meta construction only: never a fake HPU pass."""
import argparse, collections, dataclasses, json, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from gaudi_kernels.mxfp4_moe_graph import moe_historical
from gaudi_kernels.mxfp4_moe_plan import MoePlan

torch.ops.load_library(str(a.library.resolve()))
class Count(TorchDispatchMode):
    def __init__(self):super().__init__();self.calls=collections.Counter()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.calls[str(func)]+=1
        return func(*args,**(kwargs or {}))
def args(T,R,E):
    shapes=[(torch.bfloat16,(T,6144)),(torch.int32,(T,R)),(torch.float32,(T,R)),
            (torch.uint8,(E*6144,256)),(torch.uint8,(E*192,512)),
            (torch.uint8,(E*3072,256)),(torch.uint8,(E*96,512)),(torch.bfloat16,(1,512))]
    return [torch.empty(shape,dtype=dtype,device='meta') for dtype,shape in shapes]
results=[]
for T,R,E in [(2,2,2),(8,2,6),(64,8,384),(512,8,384)]:
    for mode in ['capacity_t','bucket']:
        plan=MoePlan(mode=mode,activation_budget=2*1024**3,allow_unqualified_current_graph=True,caller_certifies_fast_arithmetic=True)
        with Count() as count:
            out,status=moe_historical(*args(T,R,E),plan=plan,layout=1)
        cost=plan.costs(T,R,E)
        assert out.shape==(T,6144) and out.dtype==torch.float32 and status.shape==(1,E) and status.dtype==torch.int32
        assert count.calls['gaudi_moe_reference.count.default']==1
        assert count.calls['gaudi_moe_reference.plan.default']==1
        assert count.calls['gaudi_moe_reference.batch_mm.default']==cost['mme_nodes']
        assert count.calls['gaudi_moe_reference.gather.default']==cost['local_gp_gather_nodes']
        assert count.calls['gaudi_moe_reference.decode.default']==cost['mme_nodes']
        assert count.calls['gaudi_moe_reference.gate_rows.default']==cost['gp_gate_nodes']
        assert count.calls['gaudi_moe_reference.combine.default']==cost['down_n_tiles']
        assert all('item' not in x and '_local_scalar_dense' not in x for x in count.calls)
        results.append(dict(T=T,R=R,E=E,mode=mode,cost=cost,calls=dict(count.calls),output_shape=list(out.shape),passed=True))
rejections=[]
base=MoePlan(allow_unqualified_current_graph=True,caller_certifies_fast_arithmetic=True)
for name,inputs,plan,layout in [('M1',args(1,2,2),base,1),('wrong_layout',args(2,2,2),base,2),('no_optin',args(2,2,2),dataclasses.replace(base,allow_unqualified_current_graph=False),1),('no_arithmetic_certificate',args(2,2,2),dataclasses.replace(base,caller_certifies_fast_arithmetic=False),1),('budget',args(512,8,384),base,1)]:
    try:moe_historical(*inputs,plan=plan,layout=layout)
    except (ValueError,RuntimeError) as e:rejections.append(dict(case=name,rejected=True,error=str(e)))
    else:raise AssertionError('missing rejection '+name)
# Primitive bypass cannot silently relabel native-v2 as historical.
x=args(2,2,2);mapping=torch.empty((1,2),dtype=torch.int32,device='meta')
try:torch.ops.gaudi_moe_reference.decode(x[3],x[4],x[7],mapping,6144,512,512,0,2,0,2)
except RuntimeError as e:rejections.append(dict(case='primitive_wrong_layout',rejected=True,error=str(e)))
else:raise AssertionError('primitive layout guard absent')
report=dict(offline_meta_only=True,device_validated=False,MME_numerics_tested=False,all_pass=True,construction=results,rejections=rejections)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(all_pass=True,construction_cases=len(results),rejections=len(rejections),device_validated=False)))
