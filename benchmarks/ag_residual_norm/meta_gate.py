"""CPU-only installed-bridge metadata gate. Explicit lazy ABI before imports."""
import argparse
import json
import os
from pathlib import Path
os.environ['PT_HPU_LAZY_MODE']='1'
p=argparse.ArgumentParser();p.add_argument('--extension',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
import torch
import habana_frameworks.torch
torch.ops.load_library(str(a.extension.resolve()))
records=[]
for m,h in [(1,6144),(2,129)]:
    x=torch.empty(8*m,h,device='meta',dtype=torch.float32);r=torch.empty(m,h,device='meta',dtype=torch.bfloat16);w=torch.empty(h,device='meta',dtype=torch.bfloat16)
    for name in ['vendor_boundaries','fp32_statistics']:
        ro,y=getattr(torch.ops.gaudi_ag_norm_experiment,name)(x,r,w,1e-6)
        assert ro.shape==y.shape==r.shape and ro.dtype==y.dtype==torch.bfloat16
        records.append(dict(M=m,H=h,policy=name,output_shape=list(ro.shape)))
try:torch.ops.gaudi_ag_norm_experiment.vendor_boundaries(x.to(torch.bfloat16),r,w,1e-6)
except RuntimeError:rejected=True
else:raise AssertionError('accepted wrong dtype')
result=dict(status='CPU_META_PASS',PT_HPU_LAZY_MODE=1,device_acquired=False,records=records,wrong_dtype_rejected=rejected)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
