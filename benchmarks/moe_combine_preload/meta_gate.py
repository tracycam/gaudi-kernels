"""Exercise the installed public bridge's geometry guards without an HPU allocation."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--sha256',required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
assert hashlib.sha256(a.library.read_bytes()).hexdigest()==a.sha256
import torch
import habana_frameworks.torch
torch.ops.load_library(str(a.library.resolve()))
passed=[];rejected=[]
def values(t=1,h=6144,r=8):
    return [torch.empty((1,t*r*h),dtype=torch.float32,device='meta'),
            torch.empty((t,r),dtype=torch.float32,device='meta'),
            torch.empty((2,256),dtype=torch.uint8,device='meta'),h]
for name in ('control','rolled','unroll8'):
    op=getattr(torch.ops.gaudi_combine_preload,name)
    for t in (1,2,8):
        for h in (128,256,6144):
            y=op(*values(t,h))
            assert tuple(y.shape)==(t,h) and y.dtype==torch.float32 and y.device.type=='meta'
            passed.append(dict(op=name,t=t,h=h))
    bad={key:values()for key in ('partial_size','route_dtype','directions_shape','grad','route_count','rows','width')}
    bad['partial_size'][0]=torch.empty((1,8*6144-1),device='meta')
    bad['route_dtype'][1]=torch.empty((1,8),device='meta',dtype=torch.bfloat16)
    bad['directions_shape'][2]=torch.empty((1,256),device='meta',dtype=torch.uint8)
    bad['grad'][0].requires_grad_(True)
    bad['route_count']=values(r=7)
    bad['rows']=values(t=9)
    bad['width']=values(h=129)
    for kind,args in bad.items():
        try:op(*args)
        except RuntimeError:rejected.append(dict(op=name,case=kind))
        else:raise AssertionError((name,kind,'invalid public input accepted'))
result=dict(status='PASS_TARGET_META_INTERFACE',device_acquired=False,library_sha256=a.sha256,
            valid=passed,rejected=rejected,scope='Public Meta shape/dtype guards only; no numeric or HPU throughput claim')
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],valid=len(passed),rejected=len(rejected))))
