"""Complete residual-add + vendor RMSNorm versus one TPC node, HPU Graph replay."""
import argparse, hashlib, json, os, statistics, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--extension',required=True);p.add_argument('--quick',action='store_true');a=p.parse_args()
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from habana_frameworks.torch.hpex.normalization import FusedRMSNorm
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.residual_rmsnorm import residual_rmsnorm_bf16

torch.ops.load_library(str(Path(a.extension).resolve()));torch.manual_seed(280928);torch.set_num_threads(4)
def sync():hc.mark_step();torch.hpu.synchronize()
def vendor(x,r,w):
 z=x+r
 return z,FusedRMSNorm.apply(z,w,1e-6)
def oracle(r,w,vendor=False):
 if vendor:
  ss=(r*r).double().mean(-1,keepdim=True)
  numerator=(r*w).double()
 else:
  ss=r.double().square().mean(-1,keepdim=True)
  numerator=r.double()*w.double()
 return (numerator*torch.rsqrt(ss+float(torch.tensor(1e-6,dtype=torch.float32)))).bfloat16()
def relative(a,b):return float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30))
def maximum_ulp(a,b):
 # Distance uses sign-correct numeric values; tolerance one local BF16 ulp.
 magnitude=b.float().abs();exponent=torch.floor(torch.log2(magnitude.clamp_min(2**-126)))
 return float(((a.float()-b.float()).abs()/torch.pow(2.,exponent-7)).max())
records=[];saved={}
shapes=[(1,17),(2,129),(1,6144)] if a.quick else [(1,17),(2,129),(1,4096),(1,6144),(2,6144),(8,6144),(64,6144),(257,6144),(513,6144),(2,8192)]
for m,h in shapes:
 for pattern in (['random','cancellation','zero'] if m<=2 else ['random']):
  key=f'{m}-{h}-{pattern}';x0=torch.randn(m,h).bfloat16();r0=torch.randn(m,h).bfloat16();w0=(torch.randn(h)*.2+1).bfloat16()
  if pattern=='cancellation':r0=-x0;r0[:,::7]+=torch.tensor(.0078125,dtype=torch.bfloat16)
  if pattern=='zero':x0.zero_();r0.zero_()
  rr=(x0+r0).bfloat16();ref=(rr.double()*torch.rsqrt(rr.double().square().mean(-1,keepdim=True)+float(torch.tensor(1e-6,dtype=torch.float32)))*w0.double()).bfloat16()
  x=x0.to('hpu');r=r0.to('hpu');w=w0.to('hpu');sync()
  saved[key]={'x':x0,'residual':r0,'gamma':w0,'residual_oracle':rr,'norm_oracle_fp64_rounded':ref}
  results={}
  for name,fn in [('vendor',vendor),('fused',residual_rmsnorm_bf16)]:
   stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
   with torch.hpu.graph(graph,stream=stream):z,y=fn(x,r,w)
   sync();graph.replay(asynchronous=True);sync();actual_z=z.cpu();actual=y.cpu()
   saved[key][name+'_r']=actual_z;saved[key][name+'_y']=actual;torch.save(saved,out/'raw.pt')
   contract_ref=oracle(rr,w0,name=='vendor')
   saved[key][name+'_contract_oracle']=contract_ref
   err=relative(actual,contract_ref);ulp=maximum_ulp(actual,contract_ref)
   assert torch.equal(actual_z,rr),(key,name,'residual bits')
   assert bool(torch.isfinite(actual).all()) and err<=.003 and ulp<=1.01,(key,name,err,ulp)
   results[name]=actual
   event=[];wall=[]
   if pattern=='random':
    for _ in range(5):graph.replay(asynchronous=True)
    sync()
    for _ in range(3):
     start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True);begin=time.perf_counter();start.record(stream)
     for _ in range(100):graph.replay(asynchronous=True)
     sync();end.record(stream);end.synchronize();sync()
     event.append(start.elapsed_time(end)*1000/100);wall.append((time.perf_counter()-begin)*1e6/100)
    assert min(event)>0 and abs(statistics.median(event)-statistics.median(wall))/statistics.median(wall)<.25
   # Capture must use new input values without reallocating outputs.
   pointer=y.data_ptr();x.copy_(-x0);sync();graph.replay(asynchronous=True);sync()
   changed=y.cpu();changed_r=z.cpu();rz=(-x0+r0).bfloat16();ry=oracle(rz,w0,name=='vendor')
   saved[key][name+'_changed']=changed;torch.save(saved,out/'raw.pt')
   assert y.data_ptr()==pointer and torch.equal(changed_r,rz) and relative(changed,ry)<=.003
   x.copy_(x0);sync()
   record={'M':m,'H':h,'pattern':pattern,'route':name,'relative_l2_to_declared_contract':err,'relative_l2_to_full_fp64':relative(actual,ref),'max_bf16_ulp':ulp,'residual_bitwise':True,'event_us':event,'wall_us':wall,'output_address_stable':True}
   records.append(record);print(json.dumps(record),flush=True)
  mismatch=int((results['vendor'].view(torch.int16)!=results['fused'].view(torch.int16)).sum())
  comparison={'M':m,'H':h,'pattern':pattern,'vendor_fused_mismatch':mismatch,'vendor_fused_relative_l2':relative(results['fused'],results['vendor'])};records.append(comparison);print(json.dumps(comparison),flush=True)
(out/'result.json').write_text(json.dumps({'status':'PASS','records':records,'extension_sha256':hashlib.sha256(Path(a.extension).read_bytes()).hexdigest(),'scope':'full residual+norm graph replay; physical node audit separate; no model TPS claim'},indent=2)+'\n')
