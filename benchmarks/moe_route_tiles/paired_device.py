"""Same-input production broadcast versus experimental routed MME, public graphs.

No model default changes. Original historical uint8 owners are shared. Timings
cover producer, routing/GP/gate/down/combine and consumer, not isolated MME.
"""
import argparse,hashlib,json,os,statistics,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for n in ('fixture','runtime-fixture','core-library','metadata-library','tiles-library'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--tokens',type=int,nargs='+',default=[128,513]);p.add_argument('--rows',type=int,default=32)
p.add_argument('--replays',type=int,default=4);a=p.parse_args()
assert 1<=a.replays<=16 and len(a.tokens)==len(set(a.tokens)) and all(1<=t<=513 for t in a.tokens)
module=os.environ['GAUDI_KERNELS_MODULE_ID'];assert os.environ['HABANA_VISIBLE_MODULES']==module
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
runtime=a.runtime_fixture.resolve();plan=json.loads((runtime/'plan.json').read_text())
assert plan['format']=='gk-moe-compact8-runtime-v1'
for name,r in plan['files'].items():assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==r['sha256'],name
os.environ.update(PT_HPU_LAZY_MODE='1',PT_ENABLE_INT64_SUPPORT='0',UNIFIED_PRECISION_ROUTER='1',
                  GK_MXFP4_GP_ENABLED='0',GK_MXFP4_FOLDED_ENABLED='0',GK_MXFP4_SCALE_TAIL_ENABLED='0',GK_MXFP4_DOWN_ENABLED='0',
                  ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import numpy as np
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0,str(runtime/'executor'));import native_ops,batch_ops,precision_ops
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.mxfp4_route_tiles import moe_route_tiles
for lib in (a.core_library,a.metadata_library,a.tiles_library):torch.ops.load_library(str(lib.resolve()))
torch.set_num_threads(2)
fixture=json.loads((a.fixture/'fixture.json').read_text());assert fixture['case']=='diverse_permuted' and fixture['E']==384 and fixture['T']==513
owners=[torch.from_numpy(np.load(a.fixture/(name+'.npy'))).to('hpu')for name in ('gp','gps','down','downs')]
gp,gs,down,ds=owners;table,directions=native_ops.constants('hpu')
def sync():hc.mark_step();torch.hpu.synchronize()
def bits(x,y):return torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
result=dict(status='CHECKING',module=module,checks=[],timing=[],libraries={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()for p in (a.core_library,a.metadata_library,a.tiles_library)},
            fixture=fixture,route_state_scope='Prefix rows of frozen T513 states; for T<=384 the skew prefix is all-hot. Inspect actual IDs/counts, not the state name.',scope='Same original-width TP-local MoE owners, production broadcast vs metadata-v3 routed MME; no model or physical bandwidth/peak-utilization claim')
(out/'runtime-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
with torch.inference_mode():
 sync()
 for t in a.tokens:
  inputs={name:torch.from_numpy(np.load(a.fixture/'uniform'/(name+'.npy'))[:t].copy()).to(dtype=torch.bfloat16 if name=='x'else torch.int32 if name=='ids'else torch.float32,device='hpu')for name in ('x','ids','routing')}
  sync();graphs={};outputs={};streams={};pointers={}
  for mode in ('broadcast','routed_mme'):
   stream,graph=torch.hpu.Stream(),torch.hpu.HPUGraph()
   with torch.hpu.graph(graph,stream=stream):
    produced=inputs['x']+0
    if mode=='broadcast':y=batch_ops.moe(produced,inputs['ids'],inputs['routing'],gp,gs,down,ds,table,directions,mode='broadcast')
    else:y,_=moe_route_tiles(produced,inputs['ids'],inputs['routing'],gp,gs,down,ds,table,layout=1,rows=a.rows,n_tile=2048,metadata_version=3)
    consumer=y+.03125
   sync();graphs[mode]=graph;streams[mode]=stream;outputs[mode]=(y,consumer);pointers[mode]=[v.data_ptr()for v in outputs[mode]]
  for state in ('uniform','hot','skew','zero','restored'):
   path=a.fixture/('uniform'if state=='restored'else state)
   for name in inputs:inputs[name].copy_(torch.from_numpy(np.load(path/(name+'.npy'))[:t].copy()))
   sync();actual={}
   expected=np.load(path/'expected.npy')[:t];absolute=np.load(path/'absolute.npy')[:t]
   u=2**-24;bound=((256+8)*u/(1-(256+8)*u))*absolute+4*np.abs(np.spacing(expected)).astype(np.float64)
   for mode in graphs:
    graphs[mode].replay(asynchronous=True);sync();y,c=(v.cpu()for v in outputs[mode]);actual[mode]=y
    assert pointers[mode]==[v.data_ptr()for v in outputs[mode]]
    delta=np.abs(y.numpy().astype(np.float64)-expected);bad=int(np.count_nonzero(~np.isfinite(y.numpy())|(delta>bound)))
    check=dict(T=t,state=state,mode=mode,bad=bad,consumer_bits_equal=bits(c,y+.03125),max_abs=float(delta.max()),max_FP32_bound_ratio=float(np.max(delta/np.maximum(bound,np.finfo(np.float64).tiny))))
    torch.save(dict(y=y,consumer=c),out/f't{t}-{state}-{mode}.pt');result['checks'].append(check);save();assert bad==0 and check['consumer_bits_equal'],check
   result['checks'].append(dict(T=t,state=state,cross_mode_y_bits_equal=bits(actual['broadcast'],actual['routed_mme'])))
   if state not in ('uniform','hot','skew'):continue
   for mode in graphs:
    for _ in range(2):graphs[mode].replay(asynchronous=True)
    sync()
   for arm,mode in enumerate(('broadcast','routed_mme','routed_mme','broadcast')):
    for trial in range(3):
     start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
     stream=streams[mode]
     with torch.hpu.stream(stream):
      begin=time.perf_counter();start.record(stream)
      for _ in range(a.replays):graphs[mode].replay(asynchronous=True)
      torch.hpu.synchronize();end.record(stream);end.synchronize()
     result['timing'].append(dict(T=t,state=state,arm=arm,mode=mode,trial=trial,replays=a.replays,event_us=start.elapsed_time(end)*1000/a.replays,wall_us=(time.perf_counter()-begin)*1e6/a.replays));save()
  del graphs,outputs,streams,inputs;sync()
 result['status']='PASS_SAME_INPUT_BROADCAST_VS_ROUTED_MME';save()
