"""Review-required module5 C32/C64/C128 same-checkpoint full-chain experiment.

Full output differences versus broadcast are diagnostic. Wide variants must
additionally match C32 bitwise before timing; otherwise preserve failure for a
separate independent FP32 arithmetic audit. Never model quality acceptance.
"""
import argparse,hashlib,json,os,statistics,sys,time,traceback
from pathlib import Path
from plan import INPUT_SHA,VARIANTS,STATES,validate_runtime,schedule,complete,sha
p=argparse.ArgumentParser()
for n in('input','runtime-fixture','core-torch','metadata-torch','tiles-torch','bundle-tpc','wide-tpc','wide-torch'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--library-pins',type=Path,required=True);p.add_argument('--trials',type=int,default=3);p.add_argument('--replays',type=int,default=4);p.add_argument('--tokens',type=int,choices=(512,513),default=512);p.add_argument('--timing-anchor',choices=('broadcast','c32'),default='broadcast');p.add_argument('--routing-pattern',choices=('checkpoint','uniform_all'),default='checkpoint');a=p.parse_args()
assert 1<=a.replays<=16;arms=schedule(a.trials,a.timing_anchor)
assert os.getenv('GAUDI_KERNELS_MODULE_ID')==os.getenv('HABANA_VISIBLE_MODULES')=='5','reviewed owned module5 runner required'
assert os.getenv('PT_HPU_LAZY_MODE')=='1';assert sha(a.input)==INPUT_SHA
runtime=a.runtime_fixture.resolve();runtime_plan,registrations=validate_runtime(runtime);pins=json.loads(a.library_pins.read_text())
libs={key:getattr(a,key).resolve()for key in('core_torch','metadata_torch','tiles_torch','bundle_tpc','wide_tpc','wide_torch')}
assert set(pins)==set(libs)
for key,path in libs.items():assert sha(path)==pins[key],key
assert pins['bundle_tpc']=='11926596564549c44932360d8d13bca9724848a3dfb18146e12188827c7ad157','qualified u4/row-first bundle required'
assert pins['metadata_torch']=='639225d00ee4cb0b0f43a95d1319a3373c59ecc632889f531383a932d6506c23','qualified explicit physical-I32 metadata bridge required'
expected={str((runtime/v).resolve())for v in registrations}|{str(libs[k])for k in('bundle_tpc','wide_tpc')}|{'/usr/lib/habanalabs/libtpc_kernels.so'}
assert {str(Path(x).resolve())for x in os.getenv('GC_KERNEL_PATH','').split(':')if x}==expected
out=Path(os.environ['PROBE_OUT']);assert out.is_dir();(out/'graphs').mkdir()
os.environ.update(PT_ENABLE_INT64_SUPPORT='0',UNIFIED_PRECISION_ROUTER='1',GK_MXFP4_GP_ENABLED='0',GK_MXFP4_FOLDED_ENABLED='0',GK_MXFP4_SCALE_TAIL_ENABLED='0',GK_MXFP4_DOWN_ENABLED='0',ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
assert torch.__version__.startswith('2.11')
sys.path.insert(0,str(runtime/'executor'));import native_ops,batch_ops,precision_ops
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.mxfp4_route_tiles import moe_route_tiles
from gaudi_kernels.mxfp4_route_wide_tiles import moe_route_wide_tiles
for key in('core_torch','metadata_torch','tiles_torch','wide_torch'):torch.ops.load_library(str(libs[key]))
torch.set_num_threads(2)
def sync():hc.mark_step();torch.hpu.synchronize()
def bits(x,y):return torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
result=dict(status='CHECKING',candidate_accepted=False,model_quality_qualified=False,performance_qualified=False,source_input_sha256=INPUT_SHA,tokens=a.tokens,fixture_transform='unchanged'if a.tokens==512 else'append a copy of original last token x/ids/routing',libraries=pins,checks=[],timing=[],scope='same original checkpoint/routing owners and complete producer-MoE-consumer graphs; no quality-threshold widening or model TPS',cross_C_gate='all complete Y bits must match C32 before timing; a mismatch is not by itself proof of illegal FP32 arithmetic')
result['timing_plan']=dict(anchor=a.timing_anchor,trials=a.trials,replays=a.replays,arms=arms)
states=STATES if a.routing_pattern=='checkpoint' else (*STATES[:-1],'checkpoint_routes','hot_routes','restored')
result['states']=states;result['routing_pattern']=a.routing_pattern
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
try:
 with torch.inference_mode():
  cpu=torch.load(a.input,weights_only=False,map_location='cpu');assert cpu['x'].shape==(512,6144)and cpu['ids'].shape==(512,8)and cpu['ids'].dtype==torch.int32 and cpu['x'].dtype==torch.bfloat16
  assert cpu['routing'].dtype==torch.float32 and torch.isfinite(cpu['x']).all()and torch.isfinite(cpu['routing']).all()
  for key in('gp','gs','dp','ds'):assert cpu[key].dtype==torch.uint8
  if a.tokens==513:
   for key in('x','ids','routing'):cpu[key]=torch.cat((cpu[key],cpu[key][-1:]),0)
  checkpoint_ids=cpu['ids'].clone()
  if a.routing_pattern=='uniform_all':
   cpu['ids']=(torch.arange(a.tokens*8,dtype=torch.int32).reshape(a.tokens,8)%384).contiguous()
   assert cpu['ids'].unique().numel()==384
   result['fixture_transform']+='; replace IDs by (token*8+slot)%384; retain original x/routing and checkpoint weights'
   result['scope']='Real checkpoint weights with synthetic all-expert/hot routing, complete producer-MoE-consumer graphs; not model quality or TPS'
  fixtures={}
  for state in states:
   values={key:cpu[key].clone()for key in('x','ids','routing')}
   if state=='route_reverse':
    for key in('ids','routing'):values[key]=values[key].flip(1).contiguous()
   if state=='input_changed':values['x']=(-values['x']).roll(17,-1).contiguous()
   if state=='zero_routes':values['routing'].zero_()
   if state=='checkpoint_routes':values['ids']=checkpoint_ids.clone()
   if state=='hot_routes':values['ids']=torch.arange(8,dtype=torch.int32).expand(a.tokens,8).contiguous()
   fixtures[state]=values
  result['expert_counts']={state:torch.bincount(v['ids'].flatten().long(),minlength=384).tolist()for state,v in fixtures.items()}
  torch.save(fixtures,out/'inputs.pt')
  owners={key:value.to('hpu')for key,value in cpu.items()};sync();graphs={};outputs={};streams={};handles={};references={}
  def update(state):
   for key,value in fixtures[state].items():owners[key].copy_(value)
   sync()
  def chain(variant):
   x=owners['x']+0;ids=owners['ids']+0;routing=owners['routing']+0
   if variant=='broadcast':y=batch_ops.moe(x,ids,routing,owners['gp'],owners['gs'],owners['dp'],owners['ds'],owners['table'],owners['directions'],mode='broadcast')
   else:
    fn=moe_route_tiles if variant=='c32'else moe_route_wide_tiles
    y,metadata=fn(x.clone(),ids.to(torch.int32).clone(),routing.float().clone(),owners['gp'],owners['gs'],owners['dp'],owners['ds'],owners['table'],layout=1,rows=int(variant[1:]),n_tile=2048,metadata_version=3)
    del metadata
   return y,y+.03125
  for variant in VARIANTS:
   stream,graph=torch.hpu.Stream(),torch.hpu.HPUGraph()
   with torch.hpu.graph(graph,stream=stream):answer=chain(variant)
   sync();graphs[variant]=graph;streams[variant]=stream;outputs[variant]=answer;handles[variant]=[t.data_ptr()for t in answer]
   assert all(t.shape==(a.tokens,6144)and t.dtype==torch.float32 for t in answer)
   print('CAPTURED',variant,flush=True)
  result['logical_handles']=handles;result['input_logical_handles']={k:v.data_ptr()for k,v in owners.items()};save()
  for state in states:
   update(state);actual={}
   for variant in VARIANTS:
    samples=[]
    for replay in range(3):
     graphs[variant].replay(asynchronous=True);sync();y,c=[v.cpu()for v in outputs[variant]];samples.append(dict(y=y,consumer=c));torch.save(samples,out/f'{state}-{variant}.pt')
     finite=bool(torch.isfinite(y).all()and torch.isfinite(c).all());consumer=bits(c,y+.03125);stable=handles[variant]==[t.data_ptr()for t in outputs[variant]]
     same=not replay or bits(y,samples[0]['y']);check=dict(state=state,variant=variant,replay=replay,finite=finite,consumer_bits_equal=consumer,stable_logical_handles=stable,replay_bits_equal=same,sha256=[digest(y),digest(c)])
     result['checks'].append(check);save();assert finite and consumer and stable and same,check
     if state=='zero_routes':assert bits(y,torch.zeros_like(y))
    actual[variant]=y;references[state,variant]=(y,c)
   for variant in VARIANTS[1:]:
    y=actual[variant];old=actual['broadcast'];base=actual['c32'];delta=y.double()-old.double();cross=bits(y,base)
    result['checks'].append(dict(state=state,variant=variant,c32_bits_equal=cross,broadcast_bit_differences=int((y.view(torch.int32)!=old.view(torch.int32)).sum()),broadcast_relative_l2=float(delta.norm()/old.double().norm())if old.any()else 0.,broadcast_max_abs=float(delta.abs().max())));save()
    if not cross:raise RuntimeError('CROSS_C_FP32_AUDIT_REQUIRED: '+state+' '+variant)
  result['numeric_complete']=True;save();update('original')
  for arm in arms:
   variant=arm['variant'];graph,stream=graphs[variant],streams[variant]
   for _ in range(5):graph.replay(asynchronous=True)
   sync();start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
   with torch.hpu.stream(stream):
    begin=time.perf_counter();start.record(stream)
    for _ in range(a.replays):graph.replay(asynchronous=True)
    torch.hpu.synchronize();end.record(stream);end.synchronize()
   wall=(time.perf_counter()-begin)*1e6/a.replays;event=start.elapsed_time(end)*1000/a.replays
   y,c=[v.cpu()for v in outputs[variant]];want,consumer=references['original',variant];path=out/f"timing-{arm['pair']}-{arm['trial']}-{arm['arm']}.pt";torch.save(dict(y=y,consumer=c),path)
   record=dict(arm,event_us=event,wall_us=wall,replays=a.replays,output_bits_equal=bits(y,want),consumer_bits_equal=bits(c,consumer),stable_logical_handles=handles[variant]==[t.data_ptr()for t in outputs[variant]],output_file=path.name,sha256=[digest(y),digest(c)])
   result['timing'].append(record);save()
  assert complete(result['timing'],a.trials,a.timing_anchor)
  result['status']='PASS_CROSS_C_BITS_FULL_CHAIN_DIAGNOSTIC';result['timing_complete']=True;save()
except BaseException as e:result.update(status='FAILED_RETAINED_NOT_ACCEPTED',error=repr(e),traceback=traceback.format_exc());save();raise
