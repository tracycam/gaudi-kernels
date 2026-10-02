"""Compile actual public bridge against a CPU stub that refuses HPU execution."""
import argparse,json,os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
p=argparse.ArgumentParser();p.add_argument('--bridge-include',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);os.environ['MAX_JOBS']='1'
load(name='gaudi_down_scale_tail_meta',sources=[str(root/'csrc/torch/moe_down_scale_tail.cpp'),str(root/'benchmarks/mxfp4_moe_graph/meta_bridge_stub.cpp')],extra_include_paths=[str(a.bridge_include.resolve())],extra_cflags=['-O1'],build_directory=str(out),is_python_module=False,verbose=True)
def args(t=1,e=8):return[torch.empty(shape,dtype=dtype,device='meta')for shape,dtype in(((e*3072,256),torch.uint8),((e*96,512),torch.uint8),((t*8,256),torch.bfloat16),((1,512),torch.bfloat16),((t,8),torch.int32))]
op=torch.ops.gaudi_down_scale_tail;records=[]
for t in(1,2,8):
 for name in('control','candidate'):
  y=getattr(op,name)(*args(t));assert y.shape==(1,t*8*6144)and y.dtype==torch.float32;records.append(dict(op=name,T=t))
bad=[]
for flag in('0','1','unset'):
 if flag=='unset':os.environ.pop('PT_ENABLE_INT64_SUPPORT',None)
 else:os.environ['PT_ENABLE_INT64_SUPPORT']=flag
 for name in('control','candidate'):
  for change in('long_ids','bad_scales','strided_gate','large_t','wrong_routes'):
   x=args(9 if change=='large_t' else 1)
   if change=='long_ids':x[4]=x[4].long()
   if change=='bad_scales':x[1]=torch.empty((8*96,512),device='meta',dtype=torch.float32)
   if change=='strided_gate':x[2]=torch.empty((8,512),device='meta',dtype=torch.bfloat16)[:,::2]
   if change=='wrong_routes':x[4]=torch.empty((1,1),device='meta',dtype=torch.int32)
   try:getattr(op,name)(*x)
   except RuntimeError as exc:bad.append(dict(flag=flag,op=name,case=change,error=str(exc)))
   else:raise AssertionError('invalid argument accepted')
(out/'result.json').write_text(json.dumps(dict(status='PASS_CPU_META_ONLY',positive=records,rejections=bad,device_tested=False),indent=2)+'\n')
