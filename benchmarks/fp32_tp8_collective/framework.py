"""69 FP32 collectives inside a caller HPU graph, with the same clone producer.

Native comparison owns different recipes/buffers; do not call their difference
pure Python overhead. No borrowed native replay of framework pointers.
"""
import argparse,ctypes,json,os,sys,time
from pathlib import Path
if os.environ.get('TP8_PROBE_CPU'):os.sched_setaffinity(0,{int(os.environ['TP8_PROBE_CPU'])})
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--repeats',type=int,required=True);p.add_argument('--trials',type=int,required=True);a=p.parse_args()
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
import habana_frameworks.torch.distributed.hccl
import torch.distributed as dist
from fixtures import metric
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'benchmarks/framework'));from validation import timings_agree
rank=int(os.environ['RANK']);torch.set_num_threads(1);torch.hpu.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('hccl');assert dist.get_world_size()==8
torch.ops.load_library(str(a.library.resolve()));api=ctypes.CDLL(None);api.tp8_audit_begin.argtypes=[ctypes.c_char_p]
x=np.fromfile(a.fixture/f'input-rank{rank}.bin',np.float32).reshape(69,6144);ref=np.fromfile(a.fixture/'oracle-fp64.bin',np.float64).reshape(69,6144);ab=np.fromfile(a.fixture/'absolute-fp64.bin',np.float64).reshape(69,6144);ordered=np.fromfile(a.fixture/'ordered-fp32.bin',np.float32).reshape(69,6144)
seeds=[torch.from_numpy(v.copy()).reshape(1,6144).to('hpu') for v in x]
def sync():hc.mark_step();torch.hpu.synchronize()
records=[];sync();graphs={}
try:
 for gather in [False,True]:
  graph=torch.hpu.HPUGraph();stream=torch.hpu.Stream();outputs=[]
  with torch.hpu.graph(graph,stream=stream):
   for seed in seeds:
    source=seed.clone()  # Explicit identical synthetic producer for both routes.
    hc.mark_step()
    if gather:
     g=torch.empty((8,6144),dtype=torch.float32,device='hpu');dist.all_gather_into_tensor(g,source);y=torch.ops.tp8_fp32_probe.sum(g)
    else:y=source;dist.all_reduce(y)
    outputs.append(y)
  sync();graphs[gather]=(graph,stream,outputs,g if gather else None)
 for phase,gather in enumerate([False,True,True,False]):
  graph,stream,outputs,last_gather=graphs[gather];tag=('ags' if gather else 'ar')+str(phase);dist.barrier();sync()
  assert api.tp8_audit_begin(str(a.out/f'rank{rank}-{tag}-api.jsonl').encode())==0
  graph.replay();sync();assert api.tp8_audit_end()==0
  trace=[json.loads(line) for line in (a.out/f'rank{rank}-{tag}-api.jsonl').read_text().splitlines()];comm=[r for r in trace if r['api'].startswith('hccl')]
  assert len(comm)==69 and all(r['api']==('hcclAllGather' if gather else 'hcclAllReduce') and r['dtype']==7 and r['count']==6144 and r['status']==0 for r in comm),comm
  def verify(suffix):
   got=np.concatenate([v.cpu().numpy() for v in outputs]);got.tofile(a.out/f'rank{rank}-{tag}{suffix}.bin');result=metric(got,ref,ab);result.update(rank=rank,stage='quality',tag=tag+suffix,ordered_FP32_bit_differences=int(np.count_nonzero(got.view(np.uint32)!=ordered.view(np.uint32))))
   if gather:
    all_inputs=np.fromfile(a.fixture/'all-inputs.bin',np.float32).reshape(8,69,6144)
    result['last_AG_original_bytes_exact']=bool(np.array_equal(last_gather.cpu().numpy().view(np.uint32),all_inputs[:,-1].view(np.uint32)));assert result['last_AG_original_bytes_exact']
   records.append(result);(a.out/f'rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
   assert result['bad']==0 and (not gather or result['ordered_FP32_bit_differences']==0),result
   assert all(np.array_equal(v.cpu().numpy(),x[i:i+1]) for i,v in enumerate(seeds))
  verify('')
  for _ in range(5):graph.replay()
  sync()
  for trial in range(a.trials):
   dist.barrier();sync();begin,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True);t=time.perf_counter();begin.record(stream)
   for _ in range(a.repeats):graph.replay(asynchronous=True)
   # The closing event must not overtake host-queued graph replay jobs.
   sync();end.record(stream);end.synchronize();sync();wall=(time.perf_counter()-t)*1e6;event=begin.elapsed_time(end)*1000;valid=timings_agree([event],[wall]);records.append(dict(stage='timing',rank=rank,phase=phase,mode='FP32_AG_sum' if gather else 'FP32_AR',trial=trial,cpu_after=ctypes.CDLL(None).sched_getcpu(),allowed_cpus=sorted(os.sched_getaffinity(0)),repeats=a.repeats,collectives=69*a.repeats,event_us=event,wall_us=wall,timing_valid=valid,common_clone_producers_per_graph=69));(a.out/f'rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));assert valid,(event,wall)
  verify('-post');(a.out/f'rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
finally:dist.destroy_process_group()
