"""Whole-recipe paired padding ablation, only under run_device_probe.

One process/one module, same live original-byte weight owner for every graph.
Padding and graph-owned truncation occur inside each measured replay. No native
borrowed addresses, persistent expanded weights or production policy change.
"""
import argparse,hashlib,json,os,statistics,sys,time
from pathlib import Path
from contract import PADS,plans,reference,assess,byte_ledger
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--torch-library',type=Path,required=True);p.add_argument('--tpc-library',type=Path,required=True);p.add_argument('--pads',type=int,nargs='+',default=list(PADS));p.add_argument('--replays',type=int,default=30);p.add_argument('--warmups',type=int,default=5);a=p.parse_args();sizes=plans(a.pads);assert 1<=a.replays<=100 and 1<=a.warmups<=20
module=os.environ.get('GAUDI_KERNELS_MODULE_ID');assert module is not None and os.environ.get('HABANA_VISIBLE_MODULES')==module
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir();os.environ.update(PT_HPU_LAZY_MODE='1',ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),DUMP_POST_GRAPHS=str(out/'post_graph.json'))
# Pin the qualified native-lanes implementation; ordinary or old decoder is not this experiment.
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(a.tpc_library)=='121173a4dca0c293db042f904cdb16d2d74ab58022a9cb2f76d9a4785b776740'
assert str(a.tpc_library.resolve())in [str(Path(x).resolve())for x in os.environ.get('GC_KERNEL_PATH','').split(':')if x]
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'));sys.path.insert(0,str(root/'benchmarks/block_fp8_framework'))
from gaudi_kernels.block_fp8 import prepare_block_fp8,linear_block_fp8
from oracle import adapted_weight
from audit import audit

torch.ops.load_library(str(a.torch_library.resolve()));torch.set_num_threads(4)
def sync():hc.mark_step();torch.hpu.synchronize()
def save():
 temp=out/'result.tmp.json';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out/'result.json')
def read_graphs():
 path=out/'post_graph.json';files=sorted(path.rglob('*.json'))if path.is_dir()else[path];graphs={}
 for f in files:
  if not f.exists():continue
  for g in json.loads(f.read_text())['graphs']:
   if any(n['engine']=='MME'for n in g['nodes']):graphs[g['name']]=g
 return graphs
manifest=json.loads((a.fixture/'manifest.json').read_text())
for name in ('activation.bin','weight.bin','scales.bin'):assert sha(a.fixture/name)==manifest[name]
x0=torch.frombuffer(bytearray((a.fixture/'activation.bin').read_bytes()),dtype=torch.bfloat16).reshape(-1,6144)[:513].clone();assert x0.shape==(513,6144)
raw=torch.frombuffer(bytearray((a.fixture/'weight.bin').read_bytes()),dtype=torch.uint8).reshape(3392,6144).clone().view(torch.float8_e4m3fn)
scales=torch.frombuffer(bytearray((a.fixture/'scales.bin').read_bytes()),dtype=torch.float32).reshape(27,48).clone();bias=torch.zeros(3392)
prep=prepare_block_fp8(raw,scales,bias);decoded=adapted_weight(prep,'fp32');x1=-x0
cpu_start=time.perf_counter();refs={'initial':reference(x0,decoded,bias),'changed':reference(x1,decoded,bias)};cpu_seconds=time.perf_counter()-cpu_start
# Preserve original and adapted representation separately before any capture can fail.
torch.save(dict(x=x0,changed_x=x1,raw=raw,scales=scales,bias=bias,prepared_bytes=prep.weight.view(torch.uint8),prepared_scales=prep.scales,checkpoint_weight_sha256=prep.checkpoint_weight_sha256,rounded_half_values=prep.rounded_half_values),out/'inputs.pt')
torch.save(refs,out/'cpu-reference.pt');del decoded
result=dict(status='CHECKING',module=module,sizes=sizes,libraries={str(q.resolve()):sha(q)for q in (a.tpc_library,a.torch_library)},fixture_sha256=manifest,CPU_reference_seconds=cpu_seconds,checks=[],captures=[],windows=[],pairs=[],
    numerical_contract='Pinned native_half_rne_v1 weight representation; FP32 scale multiply then BF16 transient decode; BF16 operands with FP32 dot accumulation, FP32 bias and final BF16 RNE. Independent all-output gamma envelope, FP64 center diagnostic only.',
    measurement_scope='Full captured pad+decode+MME+finish+graph-owned split, all timed replay steps; no model TPS or physical HBM claim.')
save()
try:
 with torch.inference_mode():
  x=x0.to('hpu');w=prep.to('hpu');zeros={m:torch.zeros(m-513,6144,dtype=torch.bfloat16).to('hpu')for m in a.pads};sync();stream=torch.hpu.Stream();graphs={};owners=[]
  for m in sizes:
   before=set(read_graphs());graph=torch.hpu.HPUGraph()
   with torch.hpu.graph(graph,stream=stream):
    source=x if m==513 else torch.cat((x,zeros[m]),dim=0)
    full=linear_block_fp8(source,w,activation='bf16',scale_math='fp32')
    y=full if m==513 else torch.ops.gaudi_block_fp8.split_rows(full,513)[0]
    # Drop intermediates before capture ends, so they are not graph outputs.
    del full,source
   sync();compiled=read_graphs();new=set(compiled)-before;assert len(new)==1,('capture recipe cardinality',m,new)
   g=compiled[next(iter(new))];placement=audit(g);assert placement['no_logical_weight_read_amplification']
   ts={t['name']:t for t in g['tensors']};mm=[n for n in g['nodes']if n['engine']=='MME']
   assert all(ts[n['input_tensors'][0]]['dtype_bit_size']==16 and ts[n['input_tensors'][1]]['dtype_bit_size']==16 and ts[n['output_tensors'][0]]['dtype_bit_size']==32 for n in mm)
   effective_m=sorted({ts[n['input_tensors'][0]]['max_shape'][1]for n in mm})
   result['captures'].append(dict(padded_m=m,recipe=g['name'],effective_m=effective_m,padding_shape_preserved=effective_m==[m],placement=placement,
     mme=[{k:n[k]for k in ('name','mme_node_strategy','mme_expected_compute_cycles','mme_compute_utilization','num_of_ROIs')}for n in mm],ledger=byte_ledger(m)))
   graphs[m]=(graph,y);owners.append(y);save()
  for state,xcpu in [('initial',x0),('changed',x1)]:
   x.copy_(xcpu);sync();baseline=None
   for m in sizes:
    graph,y=graphs[m];graph.replay(asynchronous=True);sync();actual=y.cpu();torch.save(actual,out/f'{state}-m{m}.pt')
    check=assess(actual,refs[state]);check.update(state=state,padded_m=m)
    if m==513:baseline=actual
    check['baseline_BF16_bit_mismatches']=int((actual.view(torch.uint16)!=baseline.view(torch.uint16)).sum());result['checks'].append(check);save();assert check['pass_all'],check
  x.copy_(x0);sync()
  # Identical fully warmed ABBA windows for each requested padding. Baseline is
  # re-warmed in every arm as well; no fastest-sample selection or stall pruning.
  for candidate in a.pads:
   rows=[]
   for arm,m in enumerate((513,candidate,candidate,513)):
    graph,y=graphs[m]
    for _ in range(a.warmups):graph.replay(asynchronous=True)
    sync();begin,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
    start=time.perf_counter();begin.record(stream)
    for _ in range(a.replays):graph.replay(asynchronous=True)
    # Drain asynchronous submissions before end.record so it cannot overtake work.
    sync();end.record(stream);end.synchronize();sync();wall=(time.perf_counter()-start)*1e6/a.replays;event=begin.elapsed_time(end)*1000/a.replays
    window=dict(candidate=candidate,arm=arm,padded_m=m,event_us=event,wall_us=wall,replays=a.replays,warmups=a.warmups,timing_valid=event>0 and max(event,wall)/min(event,wall)<1.25)
    result['windows'].append(window);rows.append(window);save();assert window['timing_valid'],window
   av=lambda key,indices:sum(rows[i][key]for i in indices)/len(indices)
   ea,eb=av('event_us',(0,3)),av('event_us',(1,2));wa,wb=av('wall_us',(0,3)),av('wall_us',(1,2))
   result['pairs'].append(dict(padded_m=candidate,control_event_us=ea,candidate_event_us=eb,control_wall_us=wa,candidate_wall_us=wb,event_ratio=eb/ea,semantic_TFLOPS=byte_ledger(candidate)['semantic_flops']/eb/1e6,original_weight_payload_TBps=3392*6144/eb/1e6));save()
 result['status']='PASS_ALL_OUTPUTS_PLACEMENT_ABBA';save()
except BaseException as exc:
 result.update(status='FAIL',error=repr(exc));save();raise
print(json.dumps({'status':result['status'],'pairs':result['pairs']}))
