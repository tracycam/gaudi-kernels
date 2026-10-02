"""CPU rounding witnesses/oracles. Does not claim ISA or collective validation."""
import argparse
import json
from pathlib import Path
import sys
import torch
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.experimental_ag_norm import validate
torch.manual_seed(928128);torch.set_num_threads(2)

def oracle(gathered,residual,gamma,policy):
    m,h=residual.shape;s=torch.zeros(m,h,dtype=torch.float32)
    for rank in range(8):s=s+gathered[rank*m:(rank+1)*m]
    reduced=s.bfloat16();r=(reduced+residual).bfloat16()
    if policy=='vendor_boundaries':
        squares=(r*r).double();numerator=(r*gamma).double()
    else:
        squares=r.double().square();numerator=r.double()*gamma.double()
    norm=(numerator*torch.rsqrt(squares.mean(-1,keepdim=True)+float(torch.tensor(1e-6)))).bfloat16()
    return s,reduced,r,norm

report={'status':'CPU_CONTRACT_PASS_ONLY','device_used':False,'vendor_elf_obtained':False,'cases':[],'witnesses':{}}
raw={}
for m,h in [(1,1),(2,65),(2,129),(1,6144),(2,8192)]:
    x=torch.randn(8*m,h)*.125;r=torch.randn(m,h).bfloat16();w=(torch.randn(h)*.2+1).bfloat16()
    outputs={}
    for policy in ('vendor_boundaries','fp32_statistics'):
        validate(x,r,w,1e-6,policy);values=oracle(x,r,w,policy)
        assert all(torch.isfinite(v).all() for v in values)
        outputs[policy]=values
    assert torch.equal(outputs['vendor_boundaries'][2],outputs['fp32_statistics'][2])
    key=f'm{m}-h{h}';raw[key]={'gathered':x,'residual':r,'gamma':w,'outputs':outputs}
    report['cases'].append({'case':key,'norm_policy_mismatch':int((outputs['vendor_boundaries'][3]!=outputs['fp32_statistics'][3]).sum()),
                            'rank_sum_and_residual_shared':True})
# The first BF16 boundary may not be merged with residual addition.
x=torch.zeros(8,1);x[0,0]=1+2**-8;r=torch.tensor([[2**-8]],dtype=torch.bfloat16);w=torch.ones(1,dtype=torch.bfloat16)
v=oracle(x,r,w,'vendor_boundaries');wrong=(v[0]+r.float()).bfloat16()
assert not torch.equal(v[2],wrong)
report['witnesses']['merged_boundary']={'correct':v[2].float().item(),'incorrect':wrong.float().item()}
raw['merged_boundary_witness']={'gathered':x.clone(),'residual':r.clone(),'outputs':v,'incorrect_merged':wrong}
# Finite FP32 rank summation is order-sensitive; not a statement about HCCL's actual tree.
x=torch.zeros(8,1);x[:3,0]=torch.tensor([2**24,1,-2**24]);r=torch.zeros(1,1,dtype=torch.bfloat16)
v=oracle(x,r,w,'vendor_boundaries');exact=x.double().sum(0).float()
assert v[0].item()==0 and exact.item()==1
report['witnesses']['rank_order']={'sequential_fp32':v[0].item(),'fp64_then_fp32':exact.item()}
raw['rank_order_witness']={'gathered':x,'outputs':v,'fp64_then_fp32':exact}
rejects=0
for gx,rr,gg,eps,policy in [(x[:7],r,w,1e-6,'vendor_boundaries'),(x.bfloat16(),r,w,1e-6,'vendor_boundaries'),
                             (x,r,w.float(),1e-6,'vendor_boundaries'),(x,r,w,0,'vendor_boundaries'),
                             (x,r,w,1e-6,'vendor'),(x,r.unsqueeze(0),w,1e-6,'vendor_boundaries')]:
    try:validate(gx,rr,gg,eps,policy)
    except ValueError:rejects+=1
    else:raise AssertionError('invalid contract accepted')
report['rejected_metadata_cases']=rejects
a.output.mkdir(parents=True,exist_ok=False);torch.save(raw,a.output/'fixture.pt');(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
