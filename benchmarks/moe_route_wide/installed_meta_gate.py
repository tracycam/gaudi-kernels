"""Target public wide Meta schema guard, no HPU allocation/acquisition."""
import argparse,hashlib,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert hashlib.sha256(a.library.read_bytes()).hexdigest()==a.sha256
os.environ['PT_HPU_LAZY_MODE']='1'
import torch
import habana_frameworks.torch
torch.ops.load_library(str(a.library.resolve()));sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.route_wide_metadata import route_wide_metadata
make=lambda s,d=torch.int32:torch.empty(s,dtype=d,device='meta')
op=torch.ops.gaudi_route_wide_tiles;records=[]
for flag in('0','1',None):
 if flag is None:os.environ.pop('PT_ENABLE_INT64_SUPPORT',None)
 else:os.environ['PT_ENABLE_INT64_SUPPORT']=flag
 for c in(64,128):
  for t in(c,513):
   m=route_wide_metadata(make((t,8)),rows=c);assert m.rows==c and m.row_map.numel()==m.capacity*c
   x=op.gather(make((t,6144),torch.bfloat16),m.row_map,m.status,8,c,0,2);assert x.shape==(2,c,6144)
   y=op.gate(make((2,c,512),torch.float32),m.valid_rows,m.status,0);assert y.shape==(2,c,256)
   records.append(dict(flag=flag,T=t,C=c,B=m.capacity))
 errors=[]
 for fn in(lambda:route_wide_metadata(make((513,8),torch.int64),rows=64),lambda:route_wide_metadata(make((513,8)),rows=32),lambda:op.gather(make((513,6144),torch.bfloat16),make((500*129,)),make((1,)),8,129,0,2)):
  try:fn()
  except (RuntimeError,ValueError)as e:errors.append(str(e))
  else:raise AssertionError('invalid wide public input accepted')
 records.append(dict(flag=flag,rejected=errors))
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(status='PASS_INSTALLED_WIDE_META',library_sha256=a.sha256,device_acquired=False,records=records),indent=2)+'\n')
