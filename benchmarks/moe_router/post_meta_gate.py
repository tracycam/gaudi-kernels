"""CPU-only installed lazy bridge metadata gate; never creates HPU tensors."""
import argparse,json,os
from pathlib import Path
os.environ['PT_HPU_LAZY_MODE']='1'
p=argparse.ArgumentParser();p.add_argument('--extension',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
import torch
import habana_frameworks.torch
torch.ops.load_library(str(a.extension.resolve()));x=torch.empty(1,384,device='meta');ids=torch.empty(1,8,device='meta',dtype=torch.int32)
records=[]
for variant in ('scalar','vector'):
 for id_dtype in (torch.int32,torch.int64):
  for renorm,factor in [(False,1.),(True,2.5),(False,1.+2**-30)]:
   w,i,s=getattr(torch.ops.gaudi_router_post8,variant)(x,ids.to(id_dtype),factor,renorm)
   assert w.shape==i.shape==(1,8) and s.shape==(1,1) and w.dtype==s.dtype==torch.float32 and i.dtype==torch.int32
   records.append({'variant':variant,'input_id_metadata':str(id_dtype),'renormalize':renorm,'factor':factor})
rejected=0
for xx,ii in [(x.bfloat16(),ids),(x,ids.float()),(x.expand(2,384),ids),(x[:,:383],ids)]:
 try:torch.ops.gaudi_router_post8.scalar(xx,ii,2.5,True)
 except RuntimeError:rejected+=1
 else:raise AssertionError('bad metadata accepted')
result={'status':'CPU_META_PASS','device_used':False,'records':records,'rejected_metadata':rejected};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
