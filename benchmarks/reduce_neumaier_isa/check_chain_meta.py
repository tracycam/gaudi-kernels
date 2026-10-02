"""Compile current-graph Meta ops with an explicit non-executing bridge stub."""
import argparse,collections,hashlib,json,os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from torch.utils._python_dispatch import TorchDispatchMode
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--old-library',type=Path,required=True);p.add_argument('--bridge-include',required=True);p.add_argument('--synapse-include',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
os.environ.setdefault('MAX_JOBS','1')
load(name='gk_reduce_chain_offline_meta',sources=[str(root/'csrc/torch/block_fp8.cpp'),str(root/'benchmarks/mxfp4_moe_graph/meta_bridge_stub.cpp')],extra_include_paths=[a.bridge_include,a.synapse_include],extra_cflags=['-O2'],build_directory=str(a.out),is_python_module=False,verbose=True)
for path in [a.old_library,a.library]:torch.ops.load_library(str(path.resolve()))
class Count(TorchDispatchMode):
 def __init__(self):super().__init__();self.calls=collections.Counter()
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):self.calls[str(func)]+=1;return func(*args,**(kwargs or {}))
rows=[]
for m,n,k in [(1,3392,6144),(3,257,513),(16,128,128)]:
 g=(k+127)//128;x=torch.empty(m,k,dtype=torch.bfloat16,device='meta');w=torch.empty(g,n,128,dtype=torch.float8_e4m3fn,device='meta');s=torch.empty((n+127)//128,g,device='meta');b=torch.empty(n,device='meta')
 for name,op in [('old',torch.ops.gk_reduce_experiment.neumaier),('hand',torch.ops.gk_reduce_isa.handschedule)]:
  with Count() as count:
   q,sa=torch.ops.gaudi_block_fp8.quant(x);part=torch.ops.gaudi_block_fp8.batch_mm(q,w);y=op(part,sa,s,b)
  assert y.shape==(m,n) and y.dtype==torch.bfloat16
  assert sum(count.calls.values())==3,count.calls
  assert count.calls['gaudi_block_fp8.quant.default']==count.calls['gaudi_block_fp8.batch_mm.default']==1
  rows.append(dict(shape=[m,n,k],variant=name,dispatch=dict(count.calls)))
(a.out/'meta-chain.json').write_text(json.dumps(dict(all_pass=True,device_tested=False,recipe_count_not_proven_by_meta=True,rows=rows,compiled_so_sha256=hashlib.sha256((a.out/'gk_reduce_chain_offline_meta.so').read_bytes()).hexdigest()),indent=2)+'\n')
