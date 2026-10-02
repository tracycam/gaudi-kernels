"""Compile actual C++ wide+unchanged interfaces on CPU with an HPU-refusing stub."""
import argparse,json,os,sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--bridge-include',type=Path,required=True);p.add_argument('--synapse-include',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2];os.environ['MAX_JOBS']='1'
sources=[root/'csrc/torch'/s for s in('route_wide_metadata.cpp','route_wide_tiles.cpp','moe_route_metadata_v3.cpp','moe_route_tiles.cpp','mxfp4_moe_graph.cpp')]+[root/'benchmarks/mxfp4_moe_graph/meta_bridge_stub.cpp']
load(name='gk_route_wide_cpu_meta',sources=list(map(str,sources)),extra_include_paths=[str(a.bridge_include.resolve()),str(a.synapse_include.resolve())],extra_cflags=['-O1','-include',str((a.synapse_include/'synapse_common_types.h').resolve())],build_directory=str(out),is_python_module=False,verbose=True)
sys.path.insert(0,str(root/'python'));from gaudi_kernels.mxfp4_route_wide_tiles import moe_route_wide_tiles
from gaudi_kernels.route_wide_metadata import route_wide_metadata
op=torch.ops.gaudi_route_wide_tiles;meta=torch.ops.gaudi_route_metadata_wide
make=lambda shape,dt=torch.int32:torch.empty(shape,dtype=dt,device='meta')
records=[]
for t,c in ((64,64),(65,64),(128,128),(129,128),(512,64),(512,128),(513,64),(513,128)):
 e,r=384,8
 args=[make(shape,dt)for dt,shape in[(torch.bfloat16,(t,6144)),(torch.int32,(t,r)),(torch.float32,(t,r)),(torch.uint8,(e*6144,256)),(torch.uint8,(e*192,512)),(torch.uint8,(e*3072,256)),(torch.uint8,(e*96,512)),(torch.bfloat16,(1,512))]]
 y,m=moe_route_wide_tiles(*args,layout=1,rows=c,n_tile=2048)
 assert y.shape==(t,6144)and y.dtype==torch.float32 and m.rows==c
 assert m.row_map.shape==(m.capacity*c,)and m.inverse.shape==(t*r,)
 records.append(dict(T=t,C=c,B=m.capacity))
x=make((513,6144),torch.bfloat16);mapping=make((413*128,));st=make((1,));counts=make((384,));flags=make((513,));part=make((2,128,512),torch.float32);valid=make((413,))
bad=[lambda:meta.prefix(counts,flags,8,129,413),lambda:meta.prefix(counts,flags,8,128,0),lambda:meta.prefix(counts,flags,8,128,4105),lambda:meta.flatten(make((513,8),torch.int64)),lambda:route_wide_metadata(make((63,8)),rows=64),lambda:route_wide_metadata(make((513,8)),rows=32),lambda:op.gather(x,mapping,st,8,129,0,2),lambda:op.gather(x,mapping,st,8,128,413,2),lambda:op.gather(x,mapping,st,8,128,0,0),lambda:op.gate(make((2,129,512),torch.float32),valid,st,0),lambda:op.gate(part,valid,st,412),lambda:op.gate(part,valid.to(torch.int64),st,0),lambda:torch.ops.gaudi_route_metadata_v3.prefix(counts,flags,8,64,442),lambda:torch.ops.gaudi_route_tiles.gather(x,mapping,st,8,128,0,2),lambda:torch.ops.gaudi_route_tiles.gate(part,valid,st,0)]
errors=[]
for fn in bad:
 try:fn()
 except (RuntimeError,ValueError)as e:errors.append(str(e))
 else:raise AssertionError('invalid/default-expanded geometry accepted')
r=dict(status='PASS_CPU_META_WIDE',device_qualified=False,records=records,rejections=errors,scope='Actual schema/meta plus full graph Python construction only; HPU execution refused by stub, no runtime ABI or placement claim.')
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(status=r['status'],cases=len(records),rejections=len(errors))))
