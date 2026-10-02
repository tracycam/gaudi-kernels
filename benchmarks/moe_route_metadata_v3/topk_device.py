"""Bounded producer TopK(Long)->explicit I32->clone metadata capture diagnostic."""
import argparse,hashlib,json,os,sys,traceback
from contextlib import nullcontext
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--torch-library',type=Path,required=True);p.add_argument('--tokens',type=int,default=8);p.add_argument('--source',choices=['topk','native-int'],default='topk');p.add_argument('--execution',choices=['graph','eager'],default='graph');p.add_argument('--model-inputs',type=Path);a=p.parse_args()
assert os.environ.get('GAUDI_KERNELS_MODULE_ID')==os.environ.get('HABANA_VISIBLE_MODULES')=='7'
out=Path(os.environ['PROBE_OUT']);os.environ['PT_HPU_LAZY_MODE']='1';os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.route_metadata_v3 import route_metadata_v3
from gaudi_kernels.route_metadata import geometry
from reference import reference
torch.set_num_threads(2);torch.ops.load_library(str(a.torch_library.resolve()))
names=('counts','row_status','prefix','tile_expert','tile_base','valid_rows','status','row_map','inverse','chunk_offsets');t=a.tokens;e=384;r=8;c,b=geometry(t,r,e,min(32,t),None)
result=dict(status='STARTING',library_sha256=hashlib.sha256(a.torch_library.read_bytes()).hexdigest(),int64_env=os.getenv('PT_ENABLE_INT64_SUPPORT','<unset>'),tokens=t,source=a.source,checks=[])
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
def sync():hc.mark_step();torch.hpu.synchronize()
def desc(x):return dict(dtype=str(x.dtype),shape=list(x.shape),strides=list(x.stride()),contiguous=x.is_contiguous(),device=str(x.device))
try:
 with torch.inference_mode():
  # Unique integer-valued scores make TopK order independent of ties or tolerance.
  scores=torch.stack([torch.randperm(e,generator=torch.Generator().manual_seed(9823+i)) for i in range(t)]).float()
  if a.model_inputs:
   saved=torch.load(a.model_inputs,map_location='cpu',weights_only=False);target=saved['ids'].int();assert target.shape==(t,r)
   scores=torch.full((t,e),-1.,dtype=torch.float32);scores.scatter_(1,target.long(),torch.arange(r,0,-1,dtype=torch.float32).expand(t,r))
   assert torch.equal(scores.topk(r,dim=-1).indices.int(),target)
   result['model_inputs']=dict(path=str(a.model_inputs),sha256=hashlib.sha256(a.model_inputs.read_bytes()).hexdigest(),ids_dtype=str(saved['ids'].dtype))
   del saved
  initial=scores.topk(r,dim=-1).indices.int();owner=(scores if a.source=='topk' else initial).to('hpu');sync()
  graph=torch.hpu.HPUGraph();stream=torch.hpu.Stream()
  with (torch.hpu.graph(graph,stream=stream)if a.execution=='graph'else nullcontext()):
   produced=owner+0
   selected=produced.topk(r,dim=-1).indices if a.source=='topk' else produced
   ids=selected.to(torch.int32).clone()
   result['python_inputs']={'selected':desc(selected),'ids':desc(ids)};save()
   meta=route_metadata_v3(ids,experts=e,rows=c)
   outputs=[getattr(meta,name)for name in names];consumed=[x+17 for x in outputs]
  sync();pointers=[x.data_ptr()for x in outputs+consumed]
  for changed in (False,True):
   state=scores.flip(-1) if changed else scores
   expected=state.topk(r,dim=-1).indices.int()
   owner.copy_(state if a.source=='topk' else expected);sync()
   if a.execution=='graph':graph.replay(asynchronous=True);sync()
   elif changed:continue
   actual=[x.cpu()for x in outputs+consumed];want=reference(expected.tolist(),e,c,b)
   torch.save(dict(expected_ids=expected,outputs=dict(zip(names,actual[:10])),consumed=actual[10:]),out/('changed.pt'if changed else'initial.pt'))
   passed=all(torch.equal(actual[i],torch.tensor(want[name],dtype=torch.int32))and torch.equal(actual[i+10],torch.tensor(want[name],dtype=torch.int32)+17)for i,name in enumerate(names))
   result['checks'].append(dict(changed=changed,all_ten_outputs_and_consumers_equal=passed,stable_addresses=pointers==[x.data_ptr()for x in outputs+consumed]));save();assert passed,result['checks'][-1]
 result['status']='PASS_TOPK_METADATA_CAPTURE'
except Exception as exc:
 result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());save();raise
finally:save()
