"""Native activation-byte, scale and complete norm→ordinary QKV gate."""
import argparse, hashlib, json, os, statistics, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--extension',required=True);p.add_argument('--linear-extension',required=True);a=p.parse_args()
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.residual_rmsnorm import residual_rmsnorm_bf16,residual_rmsnorm_per_row_a8,residual_rmsnorm_block128_a8
from gaudi_kernels.torch_linear import prepare_separable_fp8
for f in [a.extension,a.linear_extension]:torch.ops.load_library(str(Path(f).resolve()))
torch.manual_seed(280930);torch.set_num_threads(4)
def sync():hc.mark_step();torch.hpu.synchronize()
def quant_cpu(norm,block):
 if block:
  m,h=norm.shape;norm=torch.nn.functional.pad(norm,(0,(-h)%128)).view(m,-1,128).permute(1,0,2).contiguous()
 scale=norm.float().abs().amax(-1,keepdim=True).clamp_min(1e-10)*torch.tensor(1/448,dtype=torch.float32)
 native=((norm.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*.5).to(torch.float8_e4m3fn)
 return native,scale*2
records=[];saved={}
for m,h in [(1,1),(3,129),(513,6144)]:
 xmeta=torch.empty((m,h),device='meta',dtype=torch.bfloat16);wmeta=torch.empty(h,device='meta',dtype=torch.bfloat16)
 for fn,block in [(residual_rmsnorm_per_row_a8,False),(residual_rmsnorm_block128_a8,True)]:
  rmeta,qmeta,smeta=fn(xmeta,xmeta,wmeta)
  assert rmeta.shape==(m,h) and qmeta.shape==(((h+127)//128,m,128) if block else (m,h)) and smeta.shape==(((h+127)//128,m,1) if block else (m,1))
for h,eps in [(8193,1e-6),(0,1e-6),(17,0),(17,float('nan')),(17,1e-100)]:
 xm=torch.empty((2,h),device='meta',dtype=torch.bfloat16);wm=torch.empty(h,device='meta',dtype=torch.bfloat16)
 try:residual_rmsnorm_per_row_a8(xm,xm,wm,eps)
 except RuntimeError:pass
 else:raise AssertionError('invalid metadata accepted')
for m,h,n in [(3,129,67),(1,6144,3392),(8,6144,3392),(64,6144,3392),(257,6144,129),(513,6144,129)]:
 key=f'{m}-{h}-{n}';x0=torch.randn(m,h).bfloat16();r0=torch.randn(m,h).bfloat16();gamma0=(torch.randn(h)*.2+1).bfloat16()
 if h==129:
  x0[0].zero_();r0[0].zero_();r0[1]=-x0[1];r0[1,::7]+=torch.tensor(.0078125,dtype=torch.bfloat16)
 x=x0.to('hpu');r=r0.to('hpu');gamma=gamma0.to('hpu');sync()
 rr,norm=residual_rmsnorm_bf16(x,r,gamma);sync();rr0,norm0=rr.cpu(),norm.cpu()
 original=(torch.randn(n,h)*.2).to(torch.float8_e4m3fn);weight=prepare_separable_fp8(original,torch.ones(n));bias=torch.randn(n)*.01
 w=weight.weight.to('hpu');ws=weight.scale.to('hpu');b=bias.to('hpu');sync()
 saved[key]={'x':x0,'residual':r0,'gamma':gamma0,'norm_bf16':norm0,'residual_out':rr0,'weight_native':weight.weight,'weight_scale':weight.scale,'bias':bias}
 def separate():
  residual,norm=residual_rmsnorm_bf16(x,r,gamma)
  q,s=torch.ops.gaudi_kernels._fp8_quant_fast(norm)
  prod=torch.ops.gaudi_kernels._fp8_mm_f32(q,w)
  y=torch.ops.gaudi_kernels._fp8_epilogue_rows(prod,s,ws,b)
  return residual,q,s,y
 def fused():
  residual,q,s=residual_rmsnorm_per_row_a8(x,r,gamma)
  prod=torch.ops.gaudi_kernels._fp8_mm_f32(q,w)
  y=torch.ops.gaudi_kernels._fp8_epilogue_rows(prod,s,ws,b)
  return residual,q,s,y
 outputs={}
 for name,fn in [('separate',separate),('fused',fused),('block128',lambda:residual_rmsnorm_block128_a8(x,r,gamma))]:
  block=name=='block128';qm,sm=quant_cpu(norm0,block)
  stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
  with torch.hpu.graph(graph,stream=stream):ys=fn()
  sync();graph.replay(asynchronous=True);sync();values=[t.cpu() for t in ys]
  saved[key][name]=values;saved[key][name+'_q_reference']=qm;saved[key][name+'_scale_reference']=sm;torch.save(saved,out/'raw.pt')
  mismatch=int((values[1].view(torch.uint8)!=qm.view(torch.uint8)).sum());scale_mismatch=int((values[2].view(torch.int32)!=sm.view(torch.int32)).sum())
  assert torch.equal(values[0],rr0) and mismatch==0 and scale_mismatch==0,(key,name,mismatch,scale_mismatch)
  error=0
  if not block:
   ref=(qm.double()@weight.weight.double().T*sm.double()*weight.scale.double()+bias.double()).bfloat16()
   error=float((values[3].double()-ref.double()).norm()/ref.double().norm())
   assert error<.001 and bool(torch.isfinite(values[3]).all())
   outputs[name]=values[3];saved[key][name+'_linear_ref']=ref
  for _ in range(5):graph.replay(asynchronous=True)
  sync();event=[];wall=[]
  for _ in range(3):
   start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True);begin=time.perf_counter();start.record(stream)
   for _ in range(80):graph.replay(asynchronous=True)
   sync();end.record(stream);end.synchronize();sync();event.append(start.elapsed_time(end)*1000/80);wall.append((time.perf_counter()-begin)*1e6/80)
  if h==129:
   pointers=[v.data_ptr() for v in ys];x.copy_(x0*.5);sync()
   changed_r,changed_norm=residual_rmsnorm_bf16(x,r,gamma);sync();changed_norm=changed_norm.cpu();changed_r=changed_r.cpu()
   graph.replay(asynchronous=True);sync();changed=[v.cpu() for v in ys];cq,cs=quant_cpu(changed_norm,block)
   assert pointers==[v.data_ptr() for v in ys] and torch.equal(changed[0],changed_r)
   assert torch.equal(changed[1].view(torch.uint8),cq.view(torch.uint8)) and torch.equal(changed[2].view(torch.int32),cs.view(torch.int32))
   if not block:
    cref=(cq.double()@weight.weight.double().T*cs.double()*weight.scale.double()+bias.double()).bfloat16()
    assert float((changed[3].double()-cref.double()).norm()/cref.double().norm())<.001
   saved[key][name+'_changed']=changed;x.copy_(x0);sync()
  record={'M':m,'H':h,'N':n,'route':name,'native_bytes_mismatch':mismatch,'fp32_scale_bits_mismatch':scale_mismatch,'full_gemm_relative_l2':error,'event_us':event,'wall_us':wall};records.append(record);print(json.dumps(record),flush=True)
 assert torch.equal(outputs['separate'],outputs['fused'])
 torch.save(saved,out/'raw.pt')
(out/'result.json').write_text(json.dumps({'status':'PASS','records':records,'scope':'explicit per-row fused norm→A8→ordinary FP8 GEMM; separate block128 byte/scale gate only','library_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [a.extension,a.linear_extension]}},indent=2)+'\n')
