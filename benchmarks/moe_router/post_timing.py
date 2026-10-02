"""Whole routing path ABBA; public submission/event time and trace stay separate."""
import argparse,hashlib,json,os,time
from pathlib import Path
import sys
p=argparse.ArgumentParser();p.add_argument('--extension',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--profile-only',action='store_true');p.add_argument('--factor',type=float,default=1.0);a=p.parse_args()
assert os.environ.get('GAUDI_KERNELS_MODULE_ID')=='5' and os.environ.get('HABANA_VISIBLE_MODULES')=='5'
assert hashlib.sha256(a.extension.read_bytes()).hexdigest()==a.sha256
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(PT_HPU_ENABLE_DIV_PRECISE='1',ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.experimental_router_post import post_top8
torch.ops.load_library(str(a.extension.resolve()));torch.manual_seed(928387);torch.set_num_threads(2)
def sync():hc.mark_step();torch.hpu.synchronize()
raw={};result={'status':'STARTED','profile_only':a.profile_only,'factor':a.factor,'PT_HPU_ENABLE_DIV_PRECISE':1,
 'scope':'M1 E384 top8 FP32 sigmoid+bias+vendor TopK+postprocessing+weighted consumer; no MoE weight execution/model TPS',
 'timing_scope':'public graph host enqueue/event/wall, no profiler; device-node trace measured separately',
 'records':[],'numerical':[]}
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
try:
 with torch.inference_mode():
  cpu=torch.randn(1,384);bias_cpu=torch.randn(384)*.15
  raw.update(input=cpu,bias=bias_cpu,factor=a.factor,renormalize=True);torch.save(raw,out/'fixtures.pt')
  x=cpu.to('hpu');bias=bias_cpu.to('hpu');coeff=torch.arange(1,9,dtype=torch.float32).reshape(1,8).to('hpu');sync()
  owners={}
  for mode in ('vendor','scalar','vector'):
   stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
   with torch.hpu.graph(graph,stream=stream):
    scores=x.sigmoid();ids=torch.topk(scores+bias.unsqueeze(0),8,dim=-1,sorted=False)[1]
    if mode=='vendor':
     w=scores.gather(1,ids);w=w/w.sum(-1,keepdim=True)
     if a.factor!=1.:w=w*a.factor
     ii=ids.to(torch.int32)
    else:w,ii=post_top8(scores,ids,True,a.factor,variant=mode)
    consumed=(w*coeff).sum(-1)
   owners[mode]=(stream,graph,w,ii,consumed)
  sync()
  for changed in (False,True):
   x.copy_(cpu.roll(31,-1) if changed else cpu);bias.copy_(bias_cpu.roll(47) if changed else bias_cpu);sync()
   outputs={}
   for mode,(stream,graph,*values) in owners.items():
    with torch.hpu.stream(stream):
     for _ in range(3 if a.profile_only else 10):graph.replay(asynchronous=False)
    sync();outputs[mode]=[v.cpu() for v in values]
   for mode in ('scalar','vector'):
    assert all(torch.equal(v.view(torch.int32),r.view(torch.int32)) for v,r in zip(outputs[mode],outputs['vendor'])),(mode,changed)
   raw[str(changed)]=outputs;torch.save(raw,out/'fixtures.pt')
   result['numerical'].append({'changed':changed,'weights_ids_consumer_bits_exact':True});save()
  if not a.profile_only:
   x.copy_(cpu);bias.copy_(bias_cpu);sync()
   for stream,graph,*_ in owners.values():
    with torch.hpu.stream(stream):
     for _ in range(20):graph.replay(asynchronous=False)
   sync()
   iterations=256
   for repeat in range(3):
    for candidate in ('scalar','vector'):
     for mode in ('vendor',candidate,candidate,'vendor'):
      stream,graph,*_=owners[mode];begin=torch.hpu.Event(enable_timing=True);end=torch.hpu.Event(enable_timing=True)
      sync();wall=time.perf_counter()
      with torch.hpu.stream(stream):
       begin.record(stream);enqueue=time.perf_counter()
       # False waits for host submission, not device completion. It prevents
       # a pending Python async worker from placing the end event before replay.
       for _ in range(iterations):graph.replay(asynchronous=False)
       queued=time.perf_counter();end.record(stream)
      end.synchronize();torch.hpu.synchronize();finish=time.perf_counter()
      event_us=begin.elapsed_time(end)*1000/iterations
      assert event_us>0
      result['records'].append({'repeat':repeat,'pair':candidate,'mode':mode,'iterations':iterations,
        'event_us':event_us,'wall_us':(finish-wall)*1e6/iterations,'host_enqueue_us':(queued-enqueue)*1e6/iterations});save()
  result['status']='PASS_PROFILE_ATTRIBUTION_ONLY' if a.profile_only else 'PASS_PUBLIC_ROUTER_ABBA_NO_MODEL_CLAIM'
except Exception as e:result.update(status='FAIL',error=repr(e));raise
finally:save()
