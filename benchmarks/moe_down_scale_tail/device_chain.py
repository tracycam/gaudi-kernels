"""Full production GP/gate/combine with paired down289/down242, one owned device.

All numerical states pass before timing. Events cover complete captured graphs
including bridge submission gaps. No direct-down timing or model TPS claim.
"""
import argparse,ast,json,os,re,statistics,time,traceback
from pathlib import Path
from plan import validate,sha,schedule,complete,STATES
p=argparse.ArgumentParser();p.add_argument('--runtime-fixture',type=Path,required=True);p.add_argument('--down-torch',type=Path,required=True);p.add_argument('--down-torch-sha256',required=True);p.add_argument('--rows',default='1,2,8');p.add_argument('--replays',type=int,default=16);p.add_argument('--trials',type=int,default=3);p.add_argument('--warmup',type=int,default=5);p.add_argument('--numeric-replays',type=int,default=10);a=p.parse_args()
rows=tuple(map(int,a.rows.split(',')));timing_plan=schedule(rows,a.trials)
assert 4<=a.replays<=64 and 3<=a.warmup<=20 and 1<=a.numeric_replays<=10
assert os.environ.get('GAUDI_KERNELS_MODULE_ID')==os.environ.get('HABANA_VISIBLE_MODULES')=='7','exclusive module7 runner required'
assert os.environ.get('PT_HPU_LAZY_MODE')=='1'
out=Path(os.environ['PROBE_OUT']);assert out.is_dir();runtime=a.runtime_fixture.resolve(strict=True);plan=validate(runtime)
assert re.fullmatch('[0-9a-f]{64}',a.down_torch_sha256)and sha(a.down_torch)==a.down_torch_sha256
bridge_build=json.loads((a.down_torch.parent/'build.json').read_text())
assert bridge_build['library_sha256']==a.down_torch_sha256 and bridge_build['device_acquired'] is False
assert (a.down_torch.parent/'source.cpp').read_bytes()==(Path(__file__).resolve().parents[2]/'csrc/torch/moe_down_scale_tail.cpp').read_bytes(),'bridge must match committed candidate interface'
registered={str(Path(v).resolve())for v in os.getenv('GC_KERNEL_PATH','').split(':')if v}
expected={'/usr/lib/habanalabs/libtpc_kernels.so'}|{str((runtime/plan['files'][k]['path']).resolve())for k in plan['tpc_order']};assert registered==expected,'exact five experimental/fixed databases plus vendor required'
os.environ.update(PT_ENABLE_INT64_SUPPORT='0',ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'))
(out/'graphs').mkdir();(out/'runtime-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
assert torch.__version__.startswith('2.11'),'target runtime2.11 qualification only'
torch.set_num_threads(2)
for key in plan['torch_order']:torch.ops.load_library(str(runtime/plan['files'][key]['path']))
torch.ops.load_library(str(a.down_torch.resolve()))

def frozen_function(file,name):
 source=(runtime/'source'/file).read_text();nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name==name];assert len(nodes)==1
 module=ast.Module(body=nodes,type_ignores=[]);namespace={'torch':torch};exec(compile(module,str(runtime/'source'/file),'exec'),namespace)
 (out/(name+'-executed.py')).write_text(ast.unparse(module)+'\n');return namespace[name]
constants=frozen_function('native_ops.py','constants');production_combine=frozen_function('precision_ops.py','combine')
def sync():hc.mark_step();torch.hpu.synchronize()
def same(x,y):return torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
def graph_names():
 path=out/'post_graph.json';paths=[path]if path.is_file()else list(path.rglob('*.json'))if path.is_dir()else[]
 return{g['name']for f in paths for g in json.loads(f.read_text())['graphs']}
result=dict(status='CHECKING',numeric_complete=False,timing_complete=False,rows=rows,records=[],timing=[],source_contract='Original-width E2M1/E8M0 HistoricalN512; fixed production GP scale-tail/gate/ordered FP32 combine',scope='Full TP-local graph equivalence and ABBA; no collective/model TPS or direct-kernel latency claim',event_scope='Capture-stream elapsed span including host submission gaps; synchronized wall crosscheck',down_torch_sha256=a.down_torch_sha256,down_torch_build=bridge_build,torch=torch.__version__,timing_expected_arms=len(timing_plan),weights=plan['files']['weights'],registered_TPC=sorted(registered))
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
graphs={};outputs={};streams={};owners={};fixtures={};reference={};handles={}
variants=('deployed289','wrapper289','candidate242')
def output_handles():return{k:[t.data_ptr()for t in values]for k,values in outputs.items()}
def owner_handles():return{k:[v.data_ptr()for v in values.values()]for k,values in owners.items()}
def update(n,raw):
 for key in('x','logits','gain'):owners[n][key].copy_(raw[key])
 sync()
def chain(n,variant):
 raw=owners[n];produced=raw['x']+.015625;values,ids=torch.topk(raw['logits']+.125,8,dim=-1)
 route=torch.softmax(values,-1,dtype=torch.float32)*raw['gain']
 # Retain all production materializations, including precision combine's clone.
 x=produced.clone();ids=ids.to(torch.int32).clone();route=route.to(torch.float32).clone()
 gp_partial=torch.ops.gaudi_gp_scale_tail.gp(gp,gs,x,table,ids)
 gate=torch.ops.unified_batch.gate(gp_partial,ids)
 down_op={'deployed289':torch.ops.gaudi_down_activation.broadcast,'wrapper289':torch.ops.gaudi_down_scale_tail.control,'candidate242':torch.ops.gaudi_down_scale_tail.candidate}[variant]
 partial=down_op(dp,ds,gate,table,ids)
 y=production_combine(partial,route,directions,6144,8)
 return y,y+.03125
try:
 with torch.inference_mode():
  cpu=torch.load(runtime/plan['files']['weights']['path'],map_location='cpu',weights_only=False);expected_shapes={'gp':(24*6144,256),'gs':(24*192,512),'dp':(24*3072,256),'ds':(24*96,512)}
  assert set(cpu)==set(expected_shapes)
  for key,shape in expected_shapes.items():assert cpu[key].dtype==torch.uint8 and tuple(cpu[key].shape)==shape and cpu[key].is_contiguous()
  gp,gs,dp,ds=[cpu[k].to('hpu')for k in('gp','gs','dp','ds')];table,directions=constants('hpu');sync();del cpu
  generator=torch.Generator().manual_seed(928242)
  for n in rows:
   dest=out/f'm{n}';dest.mkdir();x0=(torch.randn(n,6144,generator=generator)*.1).bfloat16();logits0=torch.randn(n,24,generator=generator);gain=torch.ones(n,1)
   hot=torch.full((n,24),-20.);hot[:,:8]=torch.arange(8,0,-1).float();cold=torch.full((n,24),-20.);cold[:,-8:]=torch.arange(8,0,-1).float()
   fixtures[n]={'initial':dict(x=x0,logits=logits0,gain=gain),'hot_changed':dict(x=(x0.roll(17,-1)*1.125).bfloat16(),logits=hot,gain=gain),'cold_changed':dict(x=-x0,logits=cold,gain=gain),'zero_routes':dict(x=x0,logits=hot,gain=torch.zeros_like(gain))}
   # Persist every original/mutated input before capture can fail.
   torch.save(fixtures[n],dest/'inputs.pt');owners[n]={k:v.to('hpu')for k,v in fixtures[n]['initial'].items()};sync()
   record=dict(rows=n,graphs=[],checks=[]);result['records'].append(record);save()
   for variant in variants:
    key=f'm{n}-{variant}';before=graph_names();stream,graph=torch.hpu.Stream(),torch.hpu.HPUGraph()
    with torch.hpu.graph(graph,stream=stream):answer=chain(n,variant)
    sync();graphs[key]=graph;streams[key]=stream;outputs[key]=answer
    assert all(t.shape==(n,6144)and t.dtype==torch.float32 for t in answer)
    record['graphs'].append(dict(variant=variant,postgraph_names=sorted(graph_names()-before)));save()
  handles={'outputs':output_handles(),'inputs':owner_handles()}
  # Complete every shape, mutation, wrapper control and replay before timing.
  for n,record in zip(rows,result['records']):
   for state,raw in fixtures[n].items():
    update(n,raw);saved=dict(inputs=raw,variants={});case=out/f'm{n}'/(state+'.pt');torch.save(saved,case)
    for variant in variants:
     key=f'm{n}-{variant}';samples=[];saved['variants'][variant]=samples
     for replay in range(a.numeric_replays):
      graphs[key].replay(asynchronous=True);sync();actual=[t.cpu()for t in outputs[key]];samples.append(actual);torch.save(saved,case)
      assert all(bool(t.isfinite().all())for t in actual),(n,state,variant,replay,'nonfinite')
      if variant=='deployed289'and replay==0:reference[(n,state)]=actual
      equal=all(same(x,y)for x,y in zip(actual,reference[(n,state)]));stable=handles=={'outputs':output_handles(),'inputs':owner_handles()}
      check=dict(state=state,variant=variant,replay=replay,output_bits_equal=equal,stable_logical_handles=stable,checked_words=sum(t.numel()for t in actual));record['checks'].append(check);save();assert equal and stable,check
      if state=='zero_routes':assert same(actual[0],torch.zeros_like(actual[0]))and same(actual[1],torch.full_like(actual[1],.03125))
  result.update(status='NUMERIC_PASS_TIMING_PENDING',numeric_complete=True);save()
  for arm in timing_plan:
   n,state,variant=arm['rows'],arm['state'],arm['variant'];key=f'm{n}-{variant}';update(n,fixtures[n][state]);graph,stream=graphs[key],streams[key]
   for _ in range(a.warmup):graph.replay(asynchronous=True)
   sync();start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
   with torch.hpu.stream(stream):
    begin=time.perf_counter();start.record(stream)
    for _ in range(a.replays):graph.replay(asynchronous=True)
    # Drain asynchronous bridge submission before recording the end event.
    torch.hpu.synchronize();end.record(stream);end.synchronize()
   wall=(time.perf_counter()-begin)*1e6/a.replays;event=start.elapsed_time(end)*1000/a.replays
   actual=[t.cpu()for t in outputs[key]];filename=f"timing-{state}-t{arm['trial']}-a{arm['arm_index']}.pt";torch.save(actual,out/f'm{n}'/filename)
   row=dict(arm,event_us=event,wall_us=wall,replays=a.replays,warmup=a.warmup,output_bits_equal=all(same(x,y)for x,y in zip(actual,reference[(n,state)])),stable_logical_handles=handles=={'outputs':output_handles(),'inputs':owner_handles()},output_file=f'm{n}/'+filename)
   result['timing'].append(row);save();assert row['output_bits_equal']and row['stable_logical_handles']and .5<event/wall<1.2,row
  assert complete(result['timing'],rows,a.trials)
  result['summary']=[dict(rows=n,state=state,variant=v,event_median_us=statistics.median(r['event_us']for r in result['timing']if(r['rows'],r['state'],r['variant'])==(n,state,v)),wall_median_us=statistics.median(r['wall_us']for r in result['timing']if(r['rows'],r['state'],r['variant'])==(n,state,v)))for n in rows for state in STATES for v in('deployed289','candidate242')]
  result.update(status='PASS_FULL_CHAIN_BITS_AND_ABBA',timing_complete=True);save()
except BaseException as exc:
 result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());save();raise
