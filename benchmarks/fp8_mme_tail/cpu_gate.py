"""CPU-only padding equality and independent FP32 envelope fault gates."""
import argparse,json
from pathlib import Path
import torch
from contract import PADS,plans,reference,assess,byte_ledger
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(48);x=torch.randn(513,64).bfloat16();w=torch.randn(7,64).bfloat16();b=torch.zeros(7,dtype=torch.float32);ref=reference(x,w,b);want=(x.float()@w.float().T).bfloat16();assert assess(want,ref)['pass_all'];records=[]
for m in PADS:
 padded=torch.cat([x,torch.zeros(m-513,64,dtype=x.dtype)]);actual=(padded.float()@w.float().T).bfloat16()[:513].contiguous()
 assert torch.equal(actual.view(torch.uint8),want.view(torch.uint8)) and assess(actual,ref)['pass_all'];records.append(byte_ledger(m))
wrong=want.clone();wrong[0,0]+=4;assert not assess(wrong,ref)['pass_all']
zero=reference(torch.zeros_like(x),w,b);assert assess(torch.zeros_like(want),zero)['pass_all'];wrongzero=torch.ones_like(want);assert not assess(wrongzero,zero)['pass_all']
for bad in [(512,),(),(528,528),(True,)]:
 try:plans(bad)
 except ValueError:pass
 else:raise AssertionError(bad)
sub=torch.zeros_like(x);sub[0,0]=2**-133
try:reference(sub,w,b)
except ValueError:pass
else:raise AssertionError('FTZ domain accepted')
a.output.write_text(json.dumps(dict(status='PASS_CPU_PADDING_AND_BOUND',records=records,fault_rejections=7,scope='Small CPU formula/control gate, no device performance or full shape arithmetic claim.'),indent=2)+'\n');print('PASS_CPU_PADDING_AND_BOUND')
