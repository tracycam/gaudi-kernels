"""Compile the real public bridge against CPU Torch with an HPU-refusing stub."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from torch.utils.cpp_extension import load

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--bridge-include',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2];os.environ['MAX_JOBS']='1'
load(name='gk_route_metadata_v3_cpu_meta',sources=[str(root/'csrc/torch/moe_route_metadata_v3.cpp'),str(root/'benchmarks/mxfp4_moe_graph/meta_bridge_stub.cpp')],extra_include_paths=[str(a.bridge_include.resolve())],extra_cflags=['-O1'],build_directory=str(a.output.resolve()),is_python_module=False,verbose=True)
sys.path.insert(0,str(root/'python'))
from gaudi_kernels.route_metadata_v3 import route_metadata_v3
records=[]
for t in (1,2,8,32,128,512,513):
 for dtype in (torch.int32,):
  result=route_metadata_v3(torch.empty((t,8),device='meta',dtype=dtype))
  assert result.row_map.shape==(result.capacity*result.rows,) and result.inverse.shape==(t*8,)
  for name in ('counts','row_status','prefix','tile_expert','tile_base','valid_rows','status','row_map','inverse','chunk_offsets'):
   x=getattr(result,name);assert x.device.type=='meta' and x.dtype==torch.int32
  records.append(dict(T=t,input_dtype=str(dtype),C=result.rows,B=result.capacity))
bad=[lambda:route_metadata_v3(torch.tensor([[2**32]],dtype=torch.int64),experts=1),
     lambda:route_metadata_v3(torch.empty((1,8),device='meta',dtype=torch.int64)),
     lambda:route_metadata_v3(torch.empty((1,8),dtype=torch.int32)),
     lambda:route_metadata_v3(torch.empty((2,16),device='meta',dtype=torch.int32)[:,::2]),
     lambda:torch.ops.gaudi_route_metadata_v3.count(torch.empty((2,8),device='meta'),torch.empty(16,device='meta',dtype=torch.int32),384),
     lambda:torch.ops.gaudi_route_metadata_v3.count(torch.empty((514,8),device='meta',dtype=torch.int32),torch.empty(4112,device='meta',dtype=torch.int32),384),
     lambda:torch.ops.gaudi_route_metadata_v3.count(torch.empty((2,8),device='meta',dtype=torch.int32),torch.empty(16,device='meta',dtype=torch.int32),2**32+384),
     lambda:torch.ops.gaudi_route_metadata_v3.prefix(torch.empty(384,device='meta',dtype=torch.int32),torch.empty(2,device='meta',dtype=torch.int32),8,1,2**32+4),
     lambda:torch.ops.gaudi_route_metadata_v3.inverse(torch.empty((2,8),device='meta',dtype=torch.int32),torch.empty(16,device='meta',dtype=torch.int32),torch.empty(385,device='meta',dtype=torch.int32),torch.empty(1,device='meta',dtype=torch.int32),torch.empty((384,2),device='meta',dtype=torch.int32),0,4),
     lambda:torch.ops.gaudi_route_metadata_v3.row_map(torch.empty(16,device='meta',dtype=torch.int32),torch.empty(4,device='meta',dtype=torch.int32),torch.empty(1,device='meta',dtype=torch.int32),2**32)]
for test in bad:
 try:test()
 except (RuntimeError,ValueError):pass
 else:raise AssertionError('invalid schema argument accepted')
result=dict(status='PASS_CPU_PUBLIC_META_V3',records=records,shape_rejections=len(bad),torch=torch.__version__,device_qualified=False,scope='Public C++ schema/meta compilation and full Python composition on Meta tensors. Stub refuses HPU execution; not installed runtime ABI/capture evidence.',files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in a.output.iterdir()if p.is_file()})
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'valid_cases':len(records),'shape_rejections':len(bad)}))
