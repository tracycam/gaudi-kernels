"""Same-resident-input full W4A16 chain, device M buckets vs broadcast.

Private operator diagnostics, no precision threshold or model TPS promotion.
Actual expert sets change between replays; debug metadata is not in timed graphs.
"""
import argparse,hashlib,json,os,statistics,sys,time,traceback
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--tokens',type=int,choices=(8,32,128,512,513),required=True)
p.add_argument('--thresholds',type=int,nargs='+',default=[64])
p.add_argument('--partition-library',type=Path,required=True)
p.add_argument('--trials',type=int,default=2)
p.add_argument('--states',nargs='+',default=['checkpoint','hot8','balanced32','mixed','permuted','zero','restored'])
a=p.parse_args();root=Path(__file__).resolve().parents[2];out=Path(os.environ['PROBE_OUT'])
assert os.environ['HABANA_VISIBLE_MODULES']==os.environ['GAUDI_KERNELS_MODULE_ID']=='5'
assert 1<=a.trials<=3 and a.thresholds and len(set(a.thresholds))==len(a.thresholds)
assert all(1<=m<=a.tokens for m in a.thresholds)
os.environ.update(PT_HPU_LAZY_MODE='1',PT_ENABLE_INT64_SUPPORT='0',UNIFIED_PRECISION_ROUTER='1',
 GK_MXFP4_GP_ENABLED='0',GK_MXFP4_FOLDED_ENABLED='0',GK_MXFP4_SCALE_TAIL_ENABLED='0',GK_MXFP4_DOWN_ENABLED='0',
 ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),DUMP_POST_GRAPHS=str(out/'post_graph.json'))
(out/'graphs').mkdir()
fixture=root/'fixtures/inputs-rank0.pt';runtime=root/'fixtures/runtime-a'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(fixture)=='6ebc2f3734fce93bf05c05fd46f8b94997f600653ab1f6860710349aa014dcca'
pins=json.loads((root/'library-pins.json').read_text())
for key,name in [('core_torch','gaudi_mxfp4_moe_graph.so'),('metadata_torch','gaudi_route_metadata_v3.so'),('tiles_torch','gaudi_route_tiles.so'),('bundle_tpc','libgaudi_route_bundle_tpc.so')]:assert sha(root/'libraries'/name)==pins[key]
snapshot=json.loads((runtime/'plan.json').read_text())
for name,record in snapshot['files'].items():assert sha(runtime/name)==record['sha256']
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0,str(runtime/'executor'))
import native_ops,batch_ops,precision_ops
sys.path.insert(0,str(root/'python'))
from gaudi_kernels.moe_expert_dispatch import expert_moe
from gaudi_kernels.moe_expert_plan import ExpertPlan,reference_partition
for name in ['gaudi_mxfp4_moe_graph.so','gaudi_route_metadata_v3.so','gaudi_route_tiles.so']:
 torch.ops.load_library(str(root/'libraries'/name))
torch.ops.load_library(str(a.partition_library.resolve()))
torch.set_num_threads(4)
def sync():hc.mark_step();torch.hpu.synchronize()
def bits(x,y):return torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
report=dict(status='running',tokens=a.tokens,thresholds=a.thresholds,checks=[],timing=[],
 scope='TP-local real checkpoint weights, full routed GP/gate/down/combine and consumer; not model or serving acceptance',
 production_default_changed=False,precision_policy='W4A16 FP32 accumulation; rounding policy deferred')
def save():(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
try:
 with torch.inference_mode():
  cpu=torch.load(fixture,map_location='cpu',weights_only=False)
  for key in ('x','ids','routing'):
   cpu[key]=cpu[key][:a.tokens].clone() if a.tokens<=512 else torch.cat((cpu[key],cpu[key][-1:]),0)
  states={}
  for state in a.states:
   values={k:cpu[k].clone() for k in ('x','ids','routing')}
   if state=='hot8':values['ids']=torch.arange(8,dtype=torch.int32).expand(a.tokens,8).contiguous()
   elif state=='balanced32':values['ids']=(torch.arange(a.tokens*8).reshape(a.tokens,8)%32).int()
   elif state=='mixed':
    values['ids'][:,:4]=torch.arange(4,dtype=torch.int32)
    values['ids'][:,4:]=(torch.arange(a.tokens*4).reshape(a.tokens,4)%380+4).int()
   elif state=='permuted':values={k:v.flip(0).contiguous() for k,v in values.items()}
   elif state=='zero':values['routing'].zero_()
   elif state not in ('checkpoint','restored'):raise ValueError('unknown input state')
   states[state]=values
  owners={k:v.to('hpu') for k,v in cpu.items()};sync()
  plans={f'm{n}':ExpertPlan(n) for n in a.thresholds};variants=('broadcast',*plans)
  report['plans']={k:{**v.costs(a.tokens,8,384),'buckets':[b.__dict__ for b in v.buckets(a.tokens,8,384)]} for k,v in plans.items()}
  graphs={};outputs={};streams={};debug_graphs={};debug_outputs={}
  def chain(variant,debug=False):
   x=owners['x'].clone();ids=owners['ids'].int().clone();routing=owners['routing'].float().clone()
   args=(x,ids,routing,owners['gp'],owners['gs'],owners['dp'],owners['ds'],owners['table'],owners['directions'])
   if variant=='broadcast':y=batch_ops.moe(*args,mode='broadcast');return y,y+.03125
   value=expert_moe(*args,plan=plans[variant],tpc=batch_ops.moe,debug=debug)
   return value if debug else (value,value+.03125)
  for variant in variants:
   graph,stream=torch.hpu.HPUGraph(),torch.hpu.Stream()
   with torch.hpu.graph(graph,stream=stream):values=chain(variant)
   sync();graphs[variant]=graph;outputs[variant]=values;streams[variant]=stream
   if variant!='broadcast':
    graph=torch.hpu.HPUGraph()
    with torch.hpu.graph(graph):values=chain(variant,True)
    sync();debug_graphs[variant]=graph;debug_outputs[variant]=values
   print('CAPTURED',variant,flush=True);save()
  baseline_checkpoint=None
  for state,inputs in states.items():
   for k,v in inputs.items():owners[k].copy_(v)
   sync();actual={}
   counts=torch.bincount(inputs['ids'].flatten().long(),minlength=384)
   for variant in variants:
    graphs[variant].replay(asynchronous=True);sync()
    y,c=(v.cpu() for v in outputs[variant]);actual[variant]=y
    check=dict(state=state,variant=variant,finite=bool(y.isfinite().all()),consumer_equal=bits(c,y+.03125),active_experts=int((counts>0).sum()),max_expert_m=int(counts.max()))
    assert check['finite'] and check['consumer_equal'],check
    if state=='zero':assert bool((y==0).all())
    metadata=[]
    if variant!='broadcast':
     debug_graphs[variant].replay(asynchronous=True);sync()
     dy,dc,meta=debug_outputs[variant]
     check['debug_y_equal']=bits(dy.cpu(),y);check['counts_equal']=torch.equal(dc.cpu(),counts.int())
     assert check['debug_y_equal'] and check['counts_equal'],check
     for item in meta:
      want=reference_partition(inputs['ids'].tolist(),384,item['bucket'])
      got={k:item[k].cpu() for k in ('experts','valid','status','inverse','mapping')}
      for key,target in [('experts','tile_expert'),('valid','valid_rows'),('status','status'),('inverse','inverse'),('mapping','row_map')]:
       assert torch.equal(got[key],torch.tensor(want[target],dtype=torch.int32)),(state,variant,key)
      metadata.append(dict(bucket=item['bucket'].__dict__,**got))
     delta=y-actual['broadcast'];check['baseline_relative_l2']=float(delta.norm()/actual['broadcast'].norm().clamp_min(1e-30));check['baseline_max_abs']=float(delta.abs().max())
    torch.save(dict(y=y,inputs=inputs,counts=counts,metadata=metadata),out/f'{state}-{variant}.pt')
    report['checks'].append(check);save()
   if state=='checkpoint':baseline_checkpoint={k:v.clone() for k,v in actual.items()}
   if state in ('restored','permuted') and baseline_checkpoint:
    for variant,y in actual.items():
     wanted=baseline_checkpoint[variant] if state=='restored' else baseline_checkpoint[variant].flip(0)
     # Same-program restored replay is exact; row permutation can change MME
     # addition organization. Report it without making bit identity a gate.
     if state=='restored':assert bits(y,wanted)
     report.setdefault('metamorphic',[]).append(dict(state=state,variant=variant,relative_l2=float((y-wanted).norm()/wanted.norm().clamp_min(1e-30))))
   if state not in ('permuted','zero','restored'):
    for variant in plans:
     for trial in range(a.trials):
      for arm,key in enumerate(('broadcast',variant,variant,'broadcast')):
       graph,stream=graphs[key],streams[key]
       for _ in range(3):graph.replay(asynchronous=True)
       sync();start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
       with torch.hpu.stream(stream):
        began=time.perf_counter_ns();start.record(stream)
        for _ in range(4):graph.replay(asynchronous=True)
        end.record(stream);end.synchronize();elapsed=(time.perf_counter_ns()-began)/4000
       y,c=(v.cpu() for v in outputs[key]);assert bits(y,actual[key]) and bits(c,y+.03125)
       report['timing'].append(dict(state=state,pair=variant,trial=trial,arm=arm,variant=key,event_us=start.elapsed_time(end)*250,wall_us=elapsed))
     save()
  report['status']='passed_functional_full_chain_diagnostic';save();print(json.dumps(report))
except BaseException as error:
 report.update(status='failed',error=repr(error),traceback=traceback.format_exc());save();raise
