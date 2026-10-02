"""Opt-in module-runner metadata gate, never invoked by the offline build.

Upstream IDs+0 and all nine downstream outputs are part of the live graph.
The caller supplies exclusive module authorization and the standard runner.
"""
import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--torch-library',type=Path,required=True);p.add_argument('--tpc-library',type=Path,required=True)
p.add_argument('--tokens',type=int,required=True);p.add_argument('--routes',type=int,default=8);p.add_argument('--experts',type=int,default=384);p.add_argument('--rows',type=int,default=16);p.add_argument('--capacity',type=int);p.add_argument('--replays',type=int,default=2);a=p.parse_args()
module=os.environ.get('GAUDI_KERNELS_MODULE_ID');assert module is not None and os.environ.get('HABANA_VISIBLE_MODULES')==module,'standard exclusive runner required'
assert 1<=a.replays<=16
out=Path(os.environ['PROBE_OUT']);assert out.is_dir()
assert str(a.tpc_library.resolve())in{str(Path(x).resolve())for x in os.getenv('GC_KERNEL_PATH','').split(':')if x}
os.environ['PT_HPU_LAZY_MODE']='1'
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),
                  GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'))
(out/'graphs').mkdir()
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.route_metadata import geometry,route_metadata
from reference import reference
torch.set_num_threads(2);torch.ops.load_library(str(a.torch_library.resolve()))
c,b=geometry(a.tokens,a.routes,a.experts,a.rows,a.capacity)
names=('counts','row_status','prefix','tile_expert','tile_base','valid_rows','status','row_map','inverse')
cases={}
for mode in ('uniform','hot','skew','bad_id','negative','duplicate'):
 if mode=='duplicate'and a.routes==1:continue
 ids=torch.tensor([[(t*a.routes+r)%a.experts for r in range(a.routes)]for t in range(a.tokens)],dtype=torch.int32)
 if mode in ('hot','skew'):
  end=a.tokens if mode=='hot'else 3*a.tokens//4
  ids[:end]=torch.arange(a.experts-a.routes,a.experts,dtype=torch.int32).flip(0)
 if mode=='bad_id':ids[0,0]=a.experts
 if mode=='negative':ids[0,0]=-1
 if mode=='duplicate':ids[0,1]=ids[0,0]
 cases[mode]=ids;torch.save(ids,out/(mode+'-ids.pt'))
result=dict(status='CHECKING',module=module,T=a.tokens,R=a.routes,E=a.experts,C=c,B=b,checks=[],
            libraries={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest()for x in (a.torch_library,a.tpc_library)},
            scope='Metadata only; no MME consumer or model-performance qualification')
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
def sync():hc.mark_step();torch.hpu.synchronize()
with torch.inference_mode():
 owner=cases['uniform'].to('hpu');sync();stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
 with torch.hpu.graph(graph,stream=stream):
  produced=owner+0
  metadata=route_metadata(produced,experts=a.experts,rows=a.rows,capacity=a.capacity)
  actual=[getattr(metadata,name)for name in names]
  consumed=[tensor+17 for tensor in actual]
 sync();pointers=[t.data_ptr()for t in actual+consumed]
 for mode,ids in cases.items():
  owner.copy_(ids);sync();graph.replay(asynchronous=True);sync()
  cpu=[x.cpu()for x in actual+consumed];want=reference(ids.tolist(),a.experts,c,b)
  torch.save(dict(zip([*names,*[n+'_consumer'for n in names]],cpu)),out/(mode+'-outputs.pt'))
  same=all(torch.equal(cpu[i],torch.tensor(want[name],dtype=torch.int32))and torch.equal(cpu[i+9],torch.tensor(want[name],dtype=torch.int32)+17)for i,name in enumerate(names))
  result['checks'].append(dict(case=mode,all_nine_outputs_and_consumers_equal=same,status=want['status'][0]));save();assert same,result['checks'][-1]
  assert pointers==[t.data_ptr()for t in actual+consumed]
 # Valid IDs restored before a bounded synchronized event/wall timing window.
 owner.copy_(cases['uniform']);sync();graph.replay(asynchronous=True);sync()
 start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
 with torch.hpu.stream(stream):
  begin=time.perf_counter();start.record(stream)
  for _ in range(a.replays):graph.replay(asynchronous=True)
  torch.hpu.synchronize();end.record(stream);end.synchronize()
 result.update(status='PASS_METADATA_CAPTURE',event_us=start.elapsed_time(end)*1000/a.replays,wall_us=(time.perf_counter()-begin)*1e6/a.replays,replays=a.replays)
 save()
