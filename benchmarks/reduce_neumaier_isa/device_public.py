"""Opt-in public current-graph gates. Execute only under run_device_probe.

The old qualified Neumaier and new GUIDs coexist; no production dispatch changes.
Raw outputs/phase summaries survive numerical or timing failure.
"""
import argparse,ctypes,hashlib,json,os,statistics,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--old-library',type=Path,required=True);p.add_argument('--block-library',type=Path,required=True);p.add_argument('--phase',choices=['reductions','chain','all'],default='all');p.add_argument('--repeats',type=int,default=30);p.add_argument('--trials',type=int,default=2);p.add_argument('--chain-case');p.add_argument('--no-timing',action='store_true');a=p.parse_args()
assert 1<=a.repeats<=100 and 1<=a.trials<=5
assert os.environ.get('GAUDI_KERNELS_MODULE_ID')=='7' and os.environ.get('HABANA_VISIBLE_MODULES')=='7','use coordinator module7 runner'
out=Path(os.environ['PROBE_OUT'])/'public';out.mkdir(parents=True,exist_ok=False);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import numpy as np
import torch
import habana_frameworks.torch.core as hc
torch.set_num_threads(4)
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'benchmarks/block_fp8_framework'))
from oracle import errors
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
identity=json.loads((a.fixtures/'fixture-identity.json').read_text())
for name,want in identity['files_sha256'].items():assert sha(a.fixtures/name)==want,name
for path in [a.old_library,a.library,a.block_library]:torch.ops.load_library(str(path.resolve()))
counter=ctypes.CDLL(None).gaudi_kernel_launch_count;counter.restype=ctypes.c_ulonglong
report=dict(status='INCOMPLETE',device_tested=True,phase=a.phase,production_default_changed=False,private_launch=False,physical_HBM_measured=False,timing_requested=not a.no_timing,timing_scope='HPUGraph host submission inclusive event and synchronized wall; not isolated kernel duration',fixture_identity_sha256=sha(a.fixtures/'fixture-identity.json'),libraries_sha256={str(p):sha(p) for p in [a.old_library,a.library,a.block_library]},numerics=[],timings=[])
def save(): (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
def sync():hc.mark_step();torch.hpu.synchronize()
def raw(t):return t.contiguous().view(torch.uint8).numpy().tobytes()
def bits(t):return np.frombuffer(raw(t),np.uint32 if t.dtype==torch.float32 else np.uint16).copy()
def capture(fn):
 stream,graph=torch.hpu.Stream(),torch.hpu.HPUGraph()
 with torch.hpu.graph(graph,stream=stream):y=fn()
 sync();graph.replay(asynchronous=True);sync();return graph,stream,y

def time_pair(case,items):
 if a.no_timing:
  # Fixed ABBA replay order supplies profiler intervals without representing
  # Python submission wall time as TPC execution time.
  for variant in ['old','hand','hand','old']:
   graph,stream,y=items[variant];before=counter()
   for _ in range(a.repeats):graph.replay(asynchronous=True)
   sync();launches=counter()-before
   report.setdefault('trace_replays',[]).append(dict(case=case,variant=variant,repeats=a.repeats,public_launches=launches))
   save();assert launches==a.repeats
  return
 for graph,stream,y in items.values():
  for _ in range(5):graph.replay(asynchronous=True)
 sync()
 for trial in range(a.trials):
  for variant in ['old','hand','hand','old']:
   graph,stream,y=items[variant];before=counter();start=time.perf_counter();begin,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True);begin.record(stream)
   for _ in range(a.repeats):graph.replay(asynchronous=True)
   sync();end.record(stream);end.synchronize();sync();wall=(time.perf_counter()-start)*1e6/a.repeats;event=begin.elapsed_time(end)*1000/a.repeats;launches=counter()-before
   valid=event>0 and event<=wall*1.1 and wall<=max(1.35*event,event+5) and launches==a.repeats
   row=dict(case=case,variant=variant,trial=trial,event_us=event,wall_us=wall,repeats=a.repeats,public_launches=launches,one_recipe_per_replay=launches==a.repeats,valid=valid);report['timings'].append(row);save()
   assert valid,row

def reductions():
 held=[]
 for case in json.loads((a.fixtures/'reduction/summary.json').read_text())['records']:
  name=case['name'];d=a.fixtures/'reduction'/name;g,m,n=case['partial_shape']
  def load(file,shape):return torch.from_numpy(np.fromfile(d/file,np.float32).reshape(shape).copy()).to('hpu')
  part=load('partial.bin',(g,m,n));sa=load('activation-scales.bin',(g,m,1));sw=load('weight-scales.bin',((n+127)//128,g));bias=load('bias.bin',(n,));sync();items={}
  ref=np.fromfile(d/'reference-fp64.bin',np.float64).reshape(m,n);bound=np.fromfile(d/'absolute-terms-fp64.bin',np.float64).reshape(m,n)
  for variant,op,fp32 in [('old',torch.ops.gk_reduce_experiment.neumaier,False),('hand',torch.ops.gk_reduce_isa.handschedule,False),('hand_debug',torch.ops.gk_reduce_isa.handschedule_debug,True)]:
   graph,stream,y=capture(lambda:op(part,sa,sw,bias));got=y.cpu();actual_bits=bits(got);expected=np.fromfile(d/('cpu-chain0-f32.bin' if fp32 else 'cpu-chain0-bf16.bin'),np.uint32 if fp32 else np.uint16)
   different=int(np.count_nonzero(actual_bits!=expected));(out/f'{name}-{variant}.bin').write_bytes(raw(got));delta=got.double().numpy()-ref;back=np.divide(np.abs(delta),bound,out=np.zeros_like(delta),where=bound!=0);back[(bound==0)&(delta!=0)]=np.inf
   rel=float(np.linalg.norm(delta)/max(np.linalg.norm(ref),1e-30));finite=bool(torch.isfinite(got).all());row=dict(phase='reduction',case=name,variant=variant,checked=m*n,output_bits=32 if fp32 else 16,bit_mismatches=different,max_abs=float(np.abs(delta).max()),relative_l2=rel,max_backward_error=float(back.max()),finite=finite,existing_adapted_fp64_gate_pass=finite and rel<.006)
   report['numerics'].append(row);save();assert different==0 and row['existing_adapted_fp64_gate_pass'],row
   if not fp32:items[variant]=(graph,stream,y)
  if name in ['qkv-measured-m1','random-m16-n3392-k6144','random-m16-n128-k128']:held.append((name,items,(part,sa,sw,bias)))
  print('NUMERIC',name,flush=True)
 for name,items,owners in held:
  time_pair(name,items)
  for variant,(_,_,y) in items.items():assert raw(y.cpu())==(out/f'{name}-{variant}.bin').read_bytes()

def chains():
 held=[]
 for case in json.loads((a.fixtures/'chain-plan.json').read_text())['records']:
  if a.chain_case and case['name']!=a.chain_case:continue
  fixture=torch.load(a.fixtures/case['file'],map_location='cpu',weights_only=False);name=case['name'];pw=fixture['weight_bits'].view(torch.float8_e4m3fn).to('hpu');sw=fixture['weight_scales'].to('hpu');bias=fixture['bias'].to('hpu');x=fixture['samples'][0]['x'].to('hpu');sync();items={};expected_by_variant={}
  q,sa=torch.ops.gaudi_block_fp8.quant(x);sync();qcpu,sacpu=q.cpu(),sa.cpu();sample=fixture['samples'][0]
  assert torch.equal(qcpu.view(torch.uint8),sample['quant_bits']) and torch.equal(sacpu,sample['activation_scales']),(name,'quant byte/scale mismatch')
  del q,sa
  for variant,reducer in [('old',torch.ops.gk_reduce_experiment.neumaier),('hand',torch.ops.gk_reduce_isa.handschedule)]:
   def apply():
    q,sa=torch.ops.gaudi_block_fp8.quant(x);partial=torch.ops.gaudi_block_fp8.batch_mm(q,pw);return reducer(partial,sa,sw,bias)
   graph,stream,y=capture(apply);items[variant]=(graph,stream,y)
  for index,sample in enumerate(fixture['samples']):
   x.copy_(sample['x']);sync();compared={}
   for variant,(graph,stream,y) in items.items():
    before=counter();graph.replay(asynchronous=True);sync();launches=counter()-before;actual=y.cpu();compared[variant]=bits(actual);(out/f'chain-{name}-{variant}-input{index}.bin').write_bytes(raw(actual));e=errors(actual,sample['refs']);gpu_error=errors(sample['refs']['gpu_style_fp64'].bfloat16(),sample['refs'])['high_fp64']['relative_l2']
    valid=e['finite'] and e['adapted_fp64']['relative_l2']<.006 and e['high_fp64']['relative_l2']<=1.15*gpu_error+.008 and launches==1
    row=dict(phase='chain',case=name,variant=variant,input_index=index,checked=actual.numel(),errors=e,existing_quality_gates_pass=valid,public_launches=launches,one_recipe_per_replay=launches==1);report['numerics'].append(row);save();assert valid,row
   equal=bool(np.array_equal(compared['old'],compared['hand']));report['numerics'].append(dict(phase='chain_pair',case=name,input_index=index,old_hand_bitwise_equal=equal));save();assert equal,(name,index,'old/hand graph mismatch')
   expected_by_variant={key:value.copy() for key,value in compared.items()}
  assert (out/f'chain-{name}-hand-input0.bin').read_bytes()!=(out/f'chain-{name}-hand-input1.bin').read_bytes()
  held.append((name,items,(pw,sw,bias,x),expected_by_variant))
 for name,items,owners,expected in held:
  time_pair('chain-'+name,items)
  for variant,(_,_,y) in items.items():assert np.array_equal(bits(y.cpu()),expected[variant]),(name,variant,'post timing changed')
try:
 save()
 if a.phase in ['all','reductions']:reductions()
 if a.phase in ['all','chain']:chains()
 report.update(status='PASS_NUMERICS_SINGLE_SUBMIT_PLACEMENT_UNAUDITED' if a.no_timing else 'PASS_NUMERICS_SINGLE_SUBMIT_TIMING_PLACEMENT_UNAUDITED',all_pass=True)
except BaseException as error:
 report.update(status='FAIL',all_pass=False,error=repr(error));raise
finally:save()
print(json.dumps(dict(status=report['status'],numerics=len(report['numerics']),timings=len(report['timings']))),flush=True)
