"""QKV graph with only production outputs retained (residual and BF16 QKV)."""
import argparse,json,os,sys,time,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--extension',required=True);p.add_argument('--linear-extension',required=True);a=p.parse_args();out=Path(os.environ['PROBE_OUT'])
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.residual_rmsnorm import residual_rmsnorm_bf16,residual_rmsnorm_per_row_a8
for path in [a.extension,a.linear_extension]:torch.ops.load_library(str(Path(path).resolve()))
torch.set_num_threads(4);torch.manual_seed(280932)
def sync():hc.mark_step();torch.hpu.synchronize()
records=[];saved={};h=6144;n=3392
w0=(torch.randn(n,h)*.1).to(torch.float8_e4m3fn);b0=torch.randn(n)*.01
w=w0.to('hpu');b=b0.to('hpu');ws=torch.ones(n).to('hpu');sync();saved['weight']=w0;saved['bias']=b0
for m in [1,64,513]:
 x0=torch.randn(m,h).bfloat16();r0=torch.randn(m,h).bfloat16();g0=(torch.randn(h)*.2+1).bfloat16()
 x=x0.to('hpu');r=r0.to('hpu');g=g0.to('hpu');sync();rr,norm=residual_rmsnorm_bf16(x,r,g);sync();rr0=rr.cpu();norm0=norm.cpu()
 scale=norm0.float().abs().amax(-1,keepdim=True).clamp_min(1e-10)*torch.tensor(1/448,dtype=torch.float32)
 q=((norm0.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*.5).to(torch.float8_e4m3fn)
 ref=(q.double()@w0.double().T*(scale.double()*2)+b0.double()).bfloat16()
 saved[str(m)]={'x':x0,'residual':r0,'gamma':g0,'norm':norm0,'reference':ref};results={}
 for fused in [False,True]:
  stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
  def chain():
   if fused:residual,qa,sa=residual_rmsnorm_per_row_a8(x,r,g)
   else:
    residual,yn=residual_rmsnorm_bf16(x,r,g)
    qa,sa=torch.ops.gaudi_kernels._fp8_quant_fast(yn)
   partial=torch.ops.gaudi_kernels._fp8_mm_f32(qa,w)
   projected=torch.ops.gaudi_kernels._fp8_epilogue_rows(partial,sa,ws,b)
   return residual,projected
  with torch.hpu.graph(graph,stream=stream):residual,projected=chain()
  sync();graph.replay(asynchronous=True);sync();actual=projected.cpu();assert torch.equal(residual.cpu(),rr0)
  err=float((actual.double()-ref.double()).norm()/ref.double().norm());assert bool(torch.isfinite(actual).all()) and err<.001
  results[fused]=actual;saved[str(m)][str(fused)]=actual
  for _ in range(5):graph.replay(asynchronous=True)
  sync();event=[];wall=[]
  for _ in range(5):
   start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True);begin=time.perf_counter();start.record(stream)
   for _ in range(100):graph.replay(asynchronous=True)
   sync();end.record(stream);end.synchronize();sync();event.append(start.elapsed_time(end)*1000/100);wall.append((time.perf_counter()-begin)*1e6/100)
  assert abs(statistics.median(event)-statistics.median(wall))/statistics.median(wall)<.1
  record={'M':m,'N':n,'K':h,'fused':fused,'full_fp64_gemm_relative_l2':err,'event_us':event,'wall_us':wall,'captured_outputs':['residual_bf16','qkv_bf16']};records.append(record);print(json.dumps(record),flush=True)
 assert torch.equal(results[False],results[True]);torch.save(saved,out/'raw.pt')
(out/'result.json').write_text(json.dumps({'status':'PASS','records':records,'scope':'complete explicit ordinary-A8 residual/norm/quant/QKV graph; no whole-model TPS'},indent=2)+'\n')
