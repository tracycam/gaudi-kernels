"""Compile actual public bridges; run Python composition on Meta, never HPU."""
import argparse,json,os,sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--bridge-include',type=Path,required=True);p.add_argument('--synapse-include',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2];os.environ['MAX_JOBS']='1'
sources=[root/'csrc/torch'/s for s in ('moe_route_tiles.cpp','moe_route_metadata.cpp','moe_route_metadata_v2.cpp','mxfp4_moe_graph.cpp')]+[root/'benchmarks/mxfp4_moe_graph/meta_bridge_stub.cpp']
load(name='gaudi_route_tiles_cpu_meta',sources=list(map(str,sources)),extra_include_paths=[str(a.bridge_include.resolve()),str(a.synapse_include.resolve())],extra_cflags=['-O1','-include',str((a.synapse_include/'synapse_common_types.h').resolve())],build_directory=str(out),is_python_module=False,verbose=True)
sys.path.insert(0,str(root/'python'));from gaudi_kernels.mxfp4_route_tiles import moe_route_tiles
op=torch.ops.gaudi_route_tiles
make=lambda shape,dtype:torch.empty(shape,dtype=dtype,device='meta')
records=[]
for t in (1,2,3,8,32,128,512,513):
 for version in (1,2):
  e,r=8,1 if t==3 else 2
  args=[make(shape,dt)for dt,shape in [(torch.bfloat16,(t,6144)),(torch.int32,(t,r)),(torch.float32,(t,r)),(torch.uint8,(e*6144,256)),(torch.uint8,(e*192,512)),(torch.uint8,(e*3072,256)),(torch.uint8,(e*96,512)),(torch.bfloat16,(1,512))]]
  y,meta=moe_route_tiles(*args,layout=1,metadata_version=version)
  assert y.shape==(t,6144)and y.dtype==torch.float32;records.append(dict(T=t,metadata=version,B=meta.capacity,C=meta.rows))
x=make((3,6144),torch.bfloat16);mapping=make((12,),torch.int32);st=make((1,),torch.int32);part=make((2,3,512),torch.float32);valid=make((4,),torch.int32);down=make((12,512),torch.float32);routing=make((3,2),torch.float32);inv=make((6,),torch.int32)
bad=[lambda:op.gather(x,mapping.to(torch.int64),st,2,3,0,2),lambda:op.gather(x,mapping,st,2**32+2,3,0,2),lambda:op.gather(x,mapping,st,2,3,3,2),lambda:op.gather(x,mapping,st,2,4,0,2),lambda:op.gather(x,mapping,st,2,3,0,0),lambda:op.gather(x,mapping,torch.empty(1,dtype=torch.int32),2,3,0,2),lambda:op.gate(part,valid.to(torch.int64),st,0),lambda:op.gate(part,valid,st,3),lambda:op.gate(part[:,:,:256],valid,st,0),lambda:op.combine(down,routing,inv.to(torch.int64),st),lambda:op.combine(down,routing,inv[:-1],st),lambda:op.combine(down[:,::2],routing,inv,st),lambda:op.reshape_i32(mapping,[2,5]),lambda:op.reshape_i32(mapping,[2**32,3]),lambda:op.reshape_i32(mapping.to(torch.int64),[3,4])]
for f in bad:
 try:f()
 except (RuntimeError,ValueError):pass
 else:raise AssertionError('invalid Meta accepted')
result=dict(status='PASS_CPU_PUBLIC_META_ROUTE_TILES',records=records,rejected=len(bad),device_qualified=False,torch=torch.__version__,scope='Actual public Meta and full fixed-capacity Python construction, not HPU ABI, graph placement, full MME math or timing.')
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='records'}))
