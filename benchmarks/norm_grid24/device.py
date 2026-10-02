"""Bounded M1 grid24 norm→current block128 QKV gate. No production patch."""
import argparse,json,os,sys,time,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--profile-only',action='store_true');a=p.parse_args()
out=Path(os.environ['PROBE_OUT']);config=json.loads(a.config.read_text())
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'))
os.environ.update({key:str(Path(value).resolve()) for key,value in config.items() if key.startswith('GK_')})
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.block_fp8 import prepare_block_fp8,linear_block_fp8_quantized
from gaudi_kernels.residual_rmsnorm import residual_rmsnorm_bf16
from gaudi_kernels.block_fp8_fp32_contract import build_reference,audit,recipe_relation
from gaudi_kernels.fp32_artifact_binding import verify_artifacts
torch.set_num_threads(4);torch.manual_seed(280964)
for key in ['GK_NORM_TORCH_LIBRARY','GK_BLOCK_FP8_TORCH_LIBRARY','GK_BLOCK_REDUCE_ISA_TORCH_LIBRARY','grid_bridge']:
 torch.ops.load_library(str(Path(config[key]).resolve()))
def sync():hc.mark_step();torch.hpu.synchronize()
def bits(x,y):return int((x.contiguous().view(torch.uint8)!=y.contiguous().view(torch.uint8)).sum())
def save(obj,name):torch.save(obj,out/name)
meta=torch.empty((1,6144),device='meta',dtype=torch.bfloat16);mg=torch.empty(6144,device='meta',dtype=torch.bfloat16)
fake=torch.ops.gaudi_norm_grid24.block128(meta,meta,mg,1e-6)
assert [tuple(t.shape)for t in fake]==[(1,6144),(48,1,128),(48,1,1)]
for shape in [(2,6144),(1,129),(1,8192)]:
 bad=torch.empty(shape,device='meta',dtype=torch.bfloat16)
 try:torch.ops.gaudi_norm_grid24.block128(bad,bad,mg,1e-6)
 except RuntimeError:pass
 else:raise AssertionError('unsupported shape accepted')
n,k=3392,6144
weight=(torch.randn(n,k)*.2).to(torch.float8_e4m3fn)
scales=(torch.rand(27,48)*.08+.01).float();bias=(torch.randn(n)*.01).float()
prepared=prepare_block_fp8(weight,scales,bias);wd=prepared.to('hpu');sync()
save(dict(weight=weight,scales=scales,bias=bias,prepared_weight=prepared.weight,prepared_scales=prepared.scales),'weights.pt')
gamma=(torch.randn(k)*.2+1).bfloat16();x0=torch.randn(1,k).bfloat16();r0=torch.randn(1,k).bfloat16()
x,r,g=[t.to('hpu')for t in [x0,r0,gamma]];sync()
reducer=torch.ops.gk_reduce_isa.handschedule
def chain(kind,diagnostic=False):
 # Real temporary producer tensors exercise the public bridge boundary.
 xx=x+.125;rr=r*.5
 if kind=='separate':
  residual,norm=residual_rmsnorm_bf16(xx,rr,g)
  q,sa=torch.ops.gaudi_block_fp8.quant(norm)
 else:residual,q,sa=torch.ops.gaudi_norm_grid24.block128(xx,rr,g,1e-6)
 witness={} if diagnostic else None
 y=linear_block_fp8_quantized(q,sa,wd,reduce_op=reducer,audit_tensors=witness)
 return (residual,y,witness) if diagnostic else (residual,y)
graphs={};records=[];timings=[]
with torch.inference_mode():
 for kind in ['separate','grid24']:
  stream,graph=torch.hpu.Stream(),torch.hpu.HPUGraph()
  with torch.hpu.graph(graph,stream=stream):values=chain(kind)
  sync();graph.replay(asynchronous=True);sync()
  graphs[kind]=(stream,graph,values,[t.data_ptr()for t in values])
 cases=['random']if a.profile_only else['random','zero','cancellation','outlier']
 for name in cases:
  xx=torch.randn(1,k).bfloat16();rr=torch.randn(1,k).bfloat16();gg=gamma.clone()
  if name=='zero':xx.fill_(-.125);rr.zero_()
  if name=='cancellation':rr=-2*(xx+.125);rr[:,::7]+=.015625
  if name=='outlier':xx[:,::128]*=128;gg[1::7]*=.001;gg[2::11]*=-1
  x.copy_(xx);r.copy_(rr);g.copy_(gg);sync()
  ref_r,ref_norm=residual_rmsnorm_bf16(x+.125,r*.5,g);sync();rn,nn=ref_r.cpu(),ref_norm.cpu();del ref_r,ref_norm
  raw=dict(x=xx,residual=rr,gamma=gg,norm=nn,residual_reference=rn)
  row=dict(fixture=name,variants={})
  expected=None if a.profile_only else build_reference(nn,weight,scales,bias=bias,native_half=True)
  for kind in ['separate','grid24']:
   stream,graph,values,ptrs=graphs[kind];graph.replay(asynchronous=True);sync();cpu=[t.cpu()for t in values]
   assert ptrs==[t.data_ptr()for t in values]and bits(cpu[0],rn)==0
   raw[kind]=cpu;row['variants'][kind]=dict(residual_bit_mismatches=bits(cpu[0],rn),output_address_stable=True)
   if not a.profile_only:
    dr,dy,diagnostic=chain(kind,True);sync();actual={key:value.cpu().contiguous()for key,value in diagnostic.items()}
    relation=recipe_relation(cpu[1],dy.cpu());assert relation['all_bits_equal'],relation
    checked=audit(expected,actual,reduction='neumaier_isa_fp32',implementation=verify_artifacts('neumaier_isa_fp32'))
    assert checked['passed'],checked
    raw[kind+'_diagnostic']=actual;row['variants'][kind].update(consumer_fp32_v1=checked,plain_vs_staged=relation)
    del dr,dy,diagnostic
  row['full_y_bits_equal']=bits(raw['separate'][1],raw['grid24'][1])==0
  assert row['full_y_bits_equal']
  if not a.profile_only:
   for field in ['q_native','activation_scales','partial']:
    assert bits(raw['separate_diagnostic'][field],raw['grid24_diagnostic'][field])==0,field
   row['actual_q_scale_partial_bits_equal']=True
  records.append(row);save(raw,name+'.pt');print(json.dumps(row),flush=True)
  for kind in ['separate','grid24','grid24','separate']:
   stream,graph,values,_=graphs[kind]
   for _ in range(3):graph.replay(asynchronous=True)
   sync();event=[];wall=[]
   for sample in range(1 if a.profile_only else 3):
    start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
    with torch.hpu.stream(stream):
     begin=time.perf_counter();start.record(stream)
     for _ in range(5 if a.profile_only else 50):graph.replay(asynchronous=True)
     torch.hpu.synchronize();end.record(stream);end.synchronize()
    count=5 if a.profile_only else 50
    event.append(start.elapsed_time(end)*1000/count);wall.append((time.perf_counter()-begin)*1e6/count)
   timings.append(dict(fixture=name,variant=kind,event_us=event,wall_us=wall))
  (out/'result.json').write_text(json.dumps(dict(status='RUNNING',records=records,abba=timings),indent=2)+'\n')
result=dict(status='PASS_M1_GRID24_QKV',records=records,abba=timings,profile_only=a.profile_only,
 scope='complete BF16 residual/FP32 norm/block128 quant/current MME+Neumaier ISA; independent synthetic checkpoint; no model installation or full producer staged certificate',
 timing_scope='full graph HPU events and synchronized wall; host supply may limit, physical engine trace required',
 requested_activation_read_bytes=589824,weight_read_change_bytes=0)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
