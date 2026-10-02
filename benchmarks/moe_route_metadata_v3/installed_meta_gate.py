"""Actual installed public schema Meta tests; no HPU tensor or acquisition."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
os.environ['PT_HPU_LAZY_MODE']='1'
import torch
import habana_frameworks.torch
torch.ops.load_library(str(a.library.resolve()))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.route_metadata_v3 import route_metadata_v3
op=torch.ops.gaudi_route_metadata_v3;records=[]
def tensor(shape,dtype=torch.int32):return torch.empty(shape,device='meta',dtype=dtype)
for flag in ('0','1',None):
 if flag is None:os.environ.pop('PT_ENABLE_INT64_SUPPORT',None)
 else:os.environ['PT_ENABLE_INT64_SUPPORT']=flag
 for t in (1,8,512,513):
  m=route_metadata_v3(tensor((t,8)))
  assert all(v.dtype==torch.int32 for v in vars(m).values()if isinstance(v,torch.Tensor))
 bad=[lambda:op.flatten(tensor((512,8),torch.int64)),
      lambda:op.count(tensor((512,8),torch.int64),tensor((4096,)),384),
      lambda:op.count(tensor((512,8)),tensor((4096,),torch.int64),384),
      lambda:op.flatten(tensor((514,8))),
      lambda:op.flatten(tensor((512,16))[:,::2]),
      lambda:op.count(tensor((512,8)),tensor((4096,)),2**32+384)]
 errors=[]
 for fn in bad:
  try:fn()
  except (ValueError,RuntimeError)as exc:errors.append(str(exc))
  else:raise AssertionError('invalid public Meta input accepted')
 records.append(dict(flag=flag,valid_shapes=4,rejections=errors))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='PASS_INSTALLED_PUBLIC_META',records=records,device_acquired=False,scope='Public Meta frontend only; backend logical-Long exception additionally requires actual HPU metadata and explicit flag0.'),indent=2)+'\n')
