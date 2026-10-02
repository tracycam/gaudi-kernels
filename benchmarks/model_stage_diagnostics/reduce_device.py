"""Opt-in public-current-graph reduction comparison; all numerics before timing."""
import argparse,hashlib,json,os,statistics,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
import numpy as np
import torch
import habana_frameworks.torch.core as hc
assert os.environ.get('HLS_MODULE_ID')=='7' and os.environ.get('GAUDI_KERNELS_MODULE_ID')=='7'
a.out.mkdir(parents=True,exist_ok=False)
fixture_identity=json.loads((a.fixtures/'fixture-identity.json').read_text())
for name,want in fixture_identity['files_sha256'].items():assert hashlib.sha256((a.fixtures/name).read_bytes()).hexdigest()==want,name
torch.ops.load_library(str(a.library.resolve()));plan=json.loads((a.fixtures/'summary.json').read_text());records=[];timings=[];held=[]
def sync():hc.mark_step();torch.hpu.synchronize()
def read(path,shape):return torch.from_numpy(np.fromfile(path,np.float32).reshape(shape).copy()).to('hpu')
for case in plan['records']:
 d=a.fixtures/case['name'];g,m,n=case['partial_shape'];part=read(d/'partial.bin',(g,m,n));sa=read(d/'activation-scales.bin',(g,m,1));sw=read(d/'weight-scales.bin',((n+127)//128,g));bias=read(d/'bias.bin',(n,));sync()
 ref=np.fromfile(d/'reference-fp64.bin',np.float64).reshape(m,n);ab=np.fromfile(d/'absolute-terms-fp64.bin',np.float64).reshape(m,n)
 for name,chain in [('original',1),('chain4',4),('neumaier',0)]:
  graph=torch.hpu.HPUGraph();stream=torch.hpu.Stream()
  with torch.hpu.graph(graph,stream=stream):y=getattr(torch.ops.gk_reduce_experiment,name)(part,sa,sw,bias)
  sync();graph.replay(asynchronous=True);sync();got=y.cpu();bits=got.view(torch.int16).numpy().view(np.uint16);expected=np.fromfile(d/f'cpu-chain{chain}-bf16.bin',np.uint16).reshape(m,n)
  assert np.array_equal(bits,expected),(case['name'],name,'declared order mismatch')
  output=a.out/(case['name']+'-'+name+'-bf16.bin');output.write_bytes(bits.tobytes());actual=got.double().numpy();delta=actual-ref;back=np.divide(np.abs(delta),ab,out=np.zeros_like(delta),where=ab!=0);back[(ab==0)&(delta!=0)]=np.inf
  rel=float(np.linalg.norm(delta)/max(np.linalg.norm(ref),1e-30));finite=bool(np.isfinite(actual).all());records.append(dict(case=case['name'],variant=name,checked=m*n,declared_cpu_bits_match=True,finite=finite,max_abs=float(np.max(np.abs(delta))),relative_l2=rel,max_backward_error=float(back.max()),bf16_mismatches=int(np.count_nonzero(bits!=torch.from_numpy(ref).bfloat16().view(torch.int16).numpy().view(np.uint16))),existing_adapted_fp64_gate_pass=finite and rel<.006))
  held.append((case,name,graph,stream,(part,sa,sw,bias,y)))
 print('NUMERIC',case['name'],flush=True)
assert all(r['existing_adapted_fp64_gate_pass'] for r in records if r['variant'] in ('original','neumaier'))
for case,name,graph,stream,owners in held:
 if case['name'] not in ['qkv-measured-m1','random-m16-n3392-k6144','random-m512-n129-k257']:continue
 for _ in range(5):graph.replay(asynchronous=True)
 sync();events=[];walls=[]
 for trial in range(5):
  begin,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True);start=time.perf_counter();begin.record(stream)
  for _ in range(50):graph.replay(asynchronous=True)
  sync();end.record(stream);end.synchronize();sync();wall=(time.perf_counter()-start)*1e6/50;event=begin.elapsed_time(end)*1000/50
  assert event>0 and event<=wall*1.1 and wall<=max(1.35*event,event+5),(event,wall)
  events.append(event);walls.append(wall)
 y=owners[-1].cpu();expected=np.fromfile(a.out/(case['name']+'-'+name+'-bf16.bin'),np.uint16).reshape(y.shape);assert np.array_equal(y.view(torch.int16).numpy().view(np.uint16),expected)
 timings.append(dict(case=case['name'],variant=name,event_us=events,wall_us=walls,median_event_us=statistics.median(events),median_wall_us=statistics.median(walls),post_timing_bits_match=True))
report=dict(fixture_identity_sha256=hashlib.sha256((a.fixtures/'fixture-identity.json').read_bytes()).hexdigest(),numerics=records,timings=timings,neumaier_all_existing_accuracy_gates_pass=True,all_variants_accuracy_gates_pass=all(r['existing_adapted_fp64_gate_pass'] for r in records),library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),production_default_changed=False,scope='standalone public HPU Graph reduction with persistent FP32 partials. Does not time original MME-to-SRAM-to-reduce pipeline or establish model quality/TPS',backward_error_denominator='sum(abs(input block partial * original activation scale * original weight scale)) + abs(bias)')
(a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(neumaier_accuracy_pass=True,all_variants_accuracy_pass=report['all_variants_accuracy_gates_pass'],timings=timings),indent=2))
