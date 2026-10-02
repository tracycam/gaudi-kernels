"""Same-live-graph API/SCAL ablation. Borrowed addresses only in bounded windows.

No owning executor claim. Native calls are gated on two identical complete
physical-binding recordings, no external events/DMA/waits/collectives, and live
owners. All framework allocation/input changes occur outside native windows.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--scope',choices=['single','chain','all'],default='all')
p.add_argument('--variants',nargs='+',choices=['existing','lut4'],default=['existing','lut4']);p.add_argument('--counter-only',action='store_true');a=p.parse_args()
assert os.environ.get('GAUDI_KERNELS_MODULE_ID')=='5'
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.torch_linear import load_extension,prepare_separable_fp8,linear_fp8
from gaudi_kernels.fp8_quantizer import prepare_fp8_quantizer
load_extension(root/os.environ['GK_TORCH_BUILD']/'gaudi_kernels_torch.so')
api=C.CDLL(None);api.submission_metric_count.restype=C.c_uint;count=api.submission_metric_count();Array=C.c_uint64*(count*3)
api.submission_snapshot.argtypes=[C.POINTER(C.c_uint64)];api.submission_metric_name.argtypes=[C.c_uint];api.submission_metric_name.restype=C.c_char_p
names=[api.submission_metric_name(i).decode() for i in range(count)]
api.submission_error.restype=C.c_char_p
api.submission_init.argtypes=[C.c_uint];api.submission_record_end.argtypes=[C.c_uint]
api.submission_dump.argtypes=[C.c_char_p]
api.submission_native.argtypes=[C.c_uint,C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)]
torch.set_num_threads(4);torch.manual_seed(270932)
plans={'lut4':prepare_fp8_quantizer('lut4').to('hpu')};owners=[plans]

def sync():hc.mark_step();torch.hpu.synchronize()
def snapshot():x=Array();api.submission_snapshot(x);return list(x)
def metric_delta(before,after):return {name:{'calls':after[3*i]-before[3*i],'host_ns':after[3*i+1]-before[3*i+1],'failures':after[3*i+2]-before[3*i+2]} for i,name in enumerate(names)}
def zero():return [0]*(count*3)
def pointers(tensors):return [(id(t),t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype),t.storage_offset()) for t in tensors]
def equal(actual,expected):return all(torch.equal(x.view(torch.int16),y.view(torch.int16)) for x,y in zip(actual,expected))
def check_oracle(actual,source,weights):
 errors=[]
 for value,w in zip(actual,weights):
  sa=source.float().abs().amax(-1,keepdim=True).clamp_min(1e-10)*(1/448)
  q=((source.float()/sa).clamp(-448,448).to(torch.float8_e4m3fn).float()*.5).to(torch.float8_e4m3fn)
  reference=((q.double()*(sa.double()*2))@w.weight.double().T*w.scale.double()+w.bias.double()).bfloat16()
  e=float((value.double()-reference.double()).norm()/reference.double().norm().clamp_min(1e-30));assert torch.isfinite(value).all() and e<.001
  errors.append(e);source=value
 return errors

def record_pair(path):
 snapshots=[]
 for compare in [0,1]:
  assert api.submission_record_begin()==0
  before=snapshot();graph.replay(asynchronous=True);sync();after=snapshot()
  rc=api.submission_record_end(compare);snapshots.append({'returncode':rc,'metrics':metric_delta(before,after)})
  api.submission_dump(str(path/f'record-{compare}.txt').encode())
  if rc:return False,api.submission_error().decode(),snapshots
 return True,'',snapshots

def native_call(repeats):
 timing=(C.c_uint64*4)();metrics=Array();rc=api.submission_native(repeats,timing,metrics)
 return {'returncode':rc,'replays':repeats,'enqueue_us':timing[0]/1000/repeats,'wall_us':timing[1]/1000/repeats,'event_us':timing[2]/1000/repeats,'unsupported_mask':timing[3],'replay_loop_metrics':metric_delta(zero(),list(metrics))}

sync();assert api.submission_init(0)==0
records=[];saved={}
shapes=[('single',512,1024,2048,1),('chain',512,2048,2048,4)]
for label,m,n,k,depth in shapes:
 if a.scope!='all' and a.scope!=label:continue
 cpu_weights=[]
 for _ in range(depth):
  raw=(torch.randn(n,k)*64).clamp(-448,448).to(torch.float8_e4m3fn)
  scales=torch.rand(n)*(.0002 if depth>1 else .02)+(.0002 if depth>1 else .01);bias=torch.randn(n)*.01
  cpu_weights.append(prepare_separable_fp8(raw,scales,bias))
 x0=torch.randn(m,k).bfloat16();xneg=x0.neg();weights=[w.to('hpu') for w in cpu_weights];x=x0.to('hpu');sync()
 saved[label]={'input':x0,'negative_input':xneg,'weights':[{'bytes':w.weight.view(torch.uint8),'scale':w.scale,'bias':w.bias} for w in cpu_weights]}
 torch.save(saved,out/'inputs-outputs.pt')  # retain fixture even if capture/gating fails
 for variant in a.variants:
  case=out/(label+'-'+variant);case.mkdir();x.copy_(x0);sync()
  stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
  with torch.hpu.graph(graph,stream=stream):
   y=x;ys=[]
   for w in weights:y=linear_fp8(y,w,activation='per_token_fp8',optimized=True,quantization=plans.get(variant));ys.append(y)
  owners.append((graph,stream,x,weights,plans,ys))
  sync()
  for _ in range(5):graph.replay(asynchronous=True)
  sync();reference=[y.cpu() for y in ys];errors=check_oracle(reference,x0,cpu_weights)
  tensor_owners=[x,*ys,*[t for w in weights for t in (w.weight,w.scale,w.bias)],plans['lut4'].table]
  row={'scope':label,'variant':variant,'shape':[m,n,k],'depth':depth,'oracle_relative_l2':errors,'owner_ids':[id(v) for v in owners[-1]],'plain_replay_async_false':None}
  # Synchronous *host submission* means all API calls return before snapshot;
  # device completion synchronization occurs after the counted replay-only window.
  before=snapshot()
  for _ in range(10):graph.replay(asynchronous=False)
  after=snapshot();sync();row['plain_replay_async_false']=metric_delta(before,after)
  py_times=[]
  for _ in range(3):
   begin,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
   sync();start=time.perf_counter();begin.record(stream);before=snapshot();enqueue=time.perf_counter()
   for _ in range(50):graph.replay(asynchronous=True)
   enqueued=time.perf_counter();sync();after=snapshot();end.record(stream);end.synchronize();sync()
   py_times.append({'replays':50,'enqueue_us':(enqueued-enqueue)*1e6/50,'wall_us':(time.perf_counter()-start)*1e6/50,'event_us':begin.elapsed_time(end)*1000/50,'replay_and_drain_metrics':metric_delta(before,after)})
  row['python']=py_times
  ready,reason,captures=record_pair(case);row.update(dual_record_stable=ready,native_accepted=ready,native_rejection=reason,captures=captures)
  if a.counter_only:
   ready=False;row.update(native_accepted=False,native_rejection=reason or 'counter-only requested; no borrowed native call attempted')
  if ready:
   rc=api.submission_find_outputs();row['output_inventory_status']=rc
   if rc:ready=False;row.update(native_accepted=False,native_rejection='compiled BF16 output inventory unsupported: '+str(rc))
  if ready:
   signature=pointers(tensor_owners);validation=[]
   for repeat in [1,10,100]:
    fresh=case/f'before-native-{repeat}';fresh.mkdir();ok,why,details=record_pair(fresh)
    assert ok,why;assert api.submission_find_outputs()==0;signature=pointers(tensor_owners)
    assert api.submission_poison_outputs()==0
    run=native_call(repeat);assert run['returncode']==0 and pointers(tensor_owners)==signature
    actual=[y.cpu() for y in ys];assert equal(actual,reference),'native output differs or poisoned outputs were not overwritten'
    run['bit_mismatches']=0;validation.append(run)
   row['native_validation']=validation
   fresh=case/'before-native-timing';fresh.mkdir();ok,why,details=record_pair(fresh)
   assert ok,why;assert api.submission_find_outputs()==0;signature=pointers(tensor_owners)
   native=[]
   for _ in range(3):
    run=native_call(50);assert run['returncode']==0;native.append(run)
   assert pointers(tensor_owners)==signature
   row['native']=native;actual=[y.cpu() for y in ys];assert equal(actual,reference)
   # Input change is outside native windows; refresh and compare physical
   # bindings twice afterwards. Poison then requires native work to restore it.
   x.copy_(xneg);sync();graph.replay(asynchronous=True);sync();changed_reference=[y.cpu() for y in ys]
   assert all(not torch.equal(v,r) for v,r in zip(changed_reference,reference))
   row['changed_oracle_relative_l2']=check_oracle(changed_reference,xneg,cpu_weights)
   changed=case/'changed';changed.mkdir();ready2,reason2,captures2=record_pair(changed)
   assert ready2,reason2;assert api.submission_find_outputs()==0;signature=pointers(tensor_owners)
   changed_runs=[]
   for repeat in [1,10,100]:
    fresh=changed/f'before-native-{repeat}';fresh.mkdir();ok,why,details=record_pair(fresh)
    assert ok,why;assert api.submission_find_outputs()==0;signature=pointers(tensor_owners)
    assert api.submission_poison_outputs()==0;run=native_call(repeat)
    assert run['returncode']==0 and pointers(tensor_owners)==signature
    actual=[y.cpu() for y in ys];assert equal(actual,changed_reference)
    run['bit_mismatches']=0;changed_runs.append(run)
   row['changed_native_validation']=changed_runs;row['changed_captures']=captures2
   saved[label][variant]={'reference':reference,'changed_reference':changed_reference,'native_changed':actual}
  else:saved[label][variant]={'reference':reference}
  row['python_median_event_us']=statistics.median(v['event_us'] for v in py_times)
  row['python_median_wall_us']=statistics.median(v['wall_us'] for v in py_times)
  if ready:
   row['native_median_event_us']=statistics.median(v['event_us'] for v in row['native']);row['native_median_wall_us']=statistics.median(v['wall_us'] for v in row['native'])
  records.append(row);(case/'result.json').write_text(json.dumps(row,indent=2)+'\n');torch.save(saved,out/'inputs-outputs.pt')
  (out/'result.json').write_text(json.dumps({'status':'INCOMPLETE','records':records},indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k in ['scope','variant','native_accepted','native_rejection','python_median_event_us','native_median_event_us','python_median_wall_us','native_median_wall_us']}),flush=True)
(out/'result.json').write_text(json.dumps({'status':'PASS' if all(r['native_accepted'] for r in records) else 'COUNTER_ONLY_NATIVE_REJECTED','native_all_pass':all(r['native_accepted'] for r in records),'records':records,'scope':'Same live framework-created recipe/address/layout, bounded compute-only borrowed-pointer ablation. Not an owning executor; API/SCAL counts are not hardware kernel/doorbell counts.'},indent=2)+'\n')
