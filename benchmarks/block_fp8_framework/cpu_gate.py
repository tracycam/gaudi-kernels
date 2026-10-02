"""CPU packing/TP/oracle gates; no HPU imports or device operations."""
import argparse
import json
from pathlib import Path
import sys
import torch

root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.block_fp8 import prepare_block_fp8,select_tp_row_group
from oracle import reference,errors
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(270928)
records=[]
for m,n,k in [(1,129,129),(3,257,513),(16,128,128)]:
    codes=torch.randint(0,256,(n,k),dtype=torch.uint8);codes[(codes&127)==127]=0
    raw=codes.view(torch.float8_e4m3fn);s=torch.rand((n+127)//128,(k+127)//128)*.001
    b=torch.randn(n)*.01;x=torch.randn(m,k).bfloat16();snapshot=codes.clone()
    w=prepare_block_fp8(raw,s,b)
    assert torch.equal(codes,snapshot)
    actual=w.weight.permute(1,0,2).contiguous().reshape(n,-1)
    assert torch.equal(actual[:,:k].view(torch.uint8),(raw.float()*.5).to(torch.float8_e4m3fn).view(torch.uint8))
    assert not bool(actual[:,k:].view(torch.uint8).any())
    assert w.weight.numel()==n*((k+127)//128)*128 and torch.equal(w.scales,s*2)
    for act,sm in [('bf16','fp32'),('bf16','bf16'),('per_block_fp8','fp32')]:
        refs=reference(x,raw,s,b,w,act,sm);e=errors(refs['adapted_fp64'].bfloat16(),refs)
        assert e['finite'];records.append({'shape':[m,n,k],'activation':act,'scale_math':sm,'errors':e,'rounded_half_values':w.rounded_half_values})
# Scale grids reset at every TP group; ceil(total N/128) is not sufficient.
raw=torch.zeros((3*129,7),dtype=torch.float8_e4m3fn);s=torch.arange(6,dtype=torch.float32).reshape(6,1)+1
w,ss=select_tp_row_group(raw,s,rank=2,group_rows=(129,129,129));assert tuple(w.shape)==(129,7) and torch.equal(ss,s[4:6])
rejected=0
for bad_w,bad_s in [(raw,s),(torch.tensor([[float('nan')]]).to(torch.float8_e4m3fn),torch.ones(1,1)),(w,ss.neg())]:
    try:prepare_block_fp8(bad_w,bad_s)
    except ValueError:rejected+=1
assert rejected==3
(a.out/'result.json').write_text(json.dumps({'status':'PASS_CPU_ONLY','records':records,'invalid_inputs_rejected':rejected},indent=2)+'\n')
print(json.dumps({'status':'PASS_CPU_ONLY','cases':len(records),'invalid_inputs_rejected':rejected}))
