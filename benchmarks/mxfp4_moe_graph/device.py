"""Small-cap HPU current-graph gate. Invoke only under run_device_probe.py."""
import argparse,ctypes,json,os,statistics,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--mode',choices=['capacity_t','bucket'],required=True);a=p.parse_args()
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),SRAM_SLICER_GRAPH_VISUALIZATION='1',DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import numpy as np
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.mxfp4_moe_graph import moe_historical
from gaudi_kernels.mxfp4_moe_plan import MoePlan
sys.path.insert(0,str(root/'benchmarks/framework'))
from validation import timings_agree
fixture=json.loads((a.fixture/'fixture.json').read_text());torch.ops.load_library(str(a.library.resolve()))
def load(name,dtype=None):
 t=torch.from_numpy(np.load(a.fixture/(name+'.npy')))
 return (t.to(dtype) if dtype is not None else t).to('hpu')
inputs=[load('x',torch.bfloat16),load('ids'),load('routing'),load('gp'),load('gps'),load('down'),load('downs'),load('lut',torch.bfloat16)]
plan=MoePlan(mode=a.mode,allow_unqualified_current_graph=True,caller_certifies_fast_arithmetic=True)
def sync():hc.mark_step();torch.hpu.synchronize()
sync();graph=torch.hpu.HPUGraph();stream=torch.hpu.Stream()
print(json.dumps(dict(stage='capture_begin',fixture=fixture,costs=plan.costs(fixture['T'],fixture['R'],fixture['E']))),flush=True)
with torch.hpu.graph(graph,stream=stream):y,status=moe_historical(*inputs,plan=plan,layout=1)
sync();graph.replay();sync()
actual=y.cpu().numpy();gotstatus=status.cpu().numpy();expected=np.load(a.fixture/'expected.npy');ab=np.load(a.fixture/'absolute.npy');delta=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
# Finite, normal certified fixtures only. Relative to sum of absolute products;
# no blanket absolute tolerance that could hide tiny outputs.
limit=3e-6*ab+4*np.spacing(np.abs(expected)).astype(np.float64)
bad=int(np.count_nonzero(~np.isfinite(actual)|(delta>limit)))+int(np.count_nonzero(gotstatus))
np.save(out/'actual.npy',actual);np.save(out/'status.npy',gotstatus)
result=dict(stage='correctness',all_pass=bad==0,bad=bad,checked=int(actual.size+gotstatus.size),max_abs=float(delta.max()),max_backward_error=float(np.max(delta/np.maximum(ab,np.finfo(np.float64).tiny))),fixture=fixture,private_native_launch=False,physical_HBM_counters_measured=False,placement_requires_audit=True)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if bad:raise RuntimeError('full-output current-graph numerical gate failed')
# Benchmark caller graph replay only after complete output gate; operator itself
# neither captures nor replays anything. A preload counter is optional but lack
# of a counter must not be called one-launch proof.
try:counter=ctypes.CDLL(None).gaudi_kernel_launch_count;counter.restype=ctypes.c_ulonglong
except AttributeError:counter=None
for _ in range(5):graph.replay()
sync();before=counter() if counter else None
for _ in range(3):graph.replay(asynchronous=True)
sync();launches=counter()-before if counter else None
if launches is not None:assert launches==3,launches
samples=[]
for _ in range(5):
 begin,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
 t=time.perf_counter();begin.record(stream)
 for _ in range(10):graph.replay(asynchronous=True)
 # Drain asynchronous host submission before enqueueing the closing event.
 # Otherwise the event can overtake replay jobs still in the host thread pool.
 sync();end.record(stream);end.synchronize();sync()
 samples.append(dict(event_us=begin.elapsed_time(end)*1000/10,wall_us=(time.perf_counter()-t)*1e6/10))
timing_valid=timings_agree([r['event_us'] for r in samples],[r['wall_us'] for r in samples])
result.update(timing=samples,timing_valid=timing_valid,event_scope='capture-stream batch, closing event after asynchronous host queue drain',event_median_us=statistics.median(r['event_us'] for r in samples),wall_median_us=statistics.median(r['wall_us'] for r in samples),synapse_launches_for_3_replays=launches)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if not timing_valid:raise RuntimeError('event/wall disagreement; timing unqualified')
