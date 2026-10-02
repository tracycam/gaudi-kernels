"""CPU FP32 arithmetic isolation on trusted SHA-sealed TP-local checkpoint data.

Not a model quality gate or device FP8 MME implementation. Each arm changes only
the named activation quantization; original MXFP4 weights and BF16 nonlinear
boundaries are retained. GPU MXFP8 reference is executed directly from pinned
local vLLM source (AST extracts one pure torch function, no GPU imports).
"""
import argparse, ast, hashlib, importlib.util, json, subprocess
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--asset', type=Path, required=True)
p.add_argument('--vllm', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--tokens', type=int, default=512)
a=p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fixture=a.asset/'fixtures/inputs-rank0.pt'
assert sha(fixture)=='6ebc2f3734fce93bf05c05fd46f8b94997f600653ab1f6860710349aa014dcca'
manifest=json.loads((a.asset/'remote-sha256.json').read_text())['files']
packing=a.asset/'fixtures/runtime-a/executor/kernels/packing.py'
assert sha(packing)==manifest[str(packing.relative_to(a.asset))]['sha256']
spec=importlib.util.spec_from_file_location('sealed_packing',packing)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
src=a.vllm/'vllm/model_executor/layers/quantization/utils/mxfp8_utils.py'
node=next(n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_mxfp8_e4m3_quantize_torch')
env=dict(torch=torch,MXFP8_BLOCK_SIZE=32,MXFP8_VALUE_DTYPE=torch.float8_e4m3fn)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),env)
gpu_quant=env[node.name]
# This file is an internally generated, SHA-pinned torch pickle, not untrusted input.
d=torch.load(fixture,map_location='cpu',weights_only=False)
rows=torch.linspace(0,511,a.tokens).round().long(); assert 8<=a.tokens<=512 and rows.unique().numel()==a.tokens
x=d['x'][rows]; ids=d['ids'][rows].long(); routes=d['routing'][rows]
lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float32)
native_rebase_stats=[]
weight_rebase_stats=[]

def weight(e,down):
 n,k=(6144,256) if down else (512,6144); nb=n//512
 w,s=(d['dp'],d['ds']) if down else (d['gp'],d['gs'])
 packed=w[e*nb*k:(e+1)*nb*k].numpy().reshape(nb,k,256)
 scales=s[e*nb*(k//32):(e+1)*nb*(k//32)].numpy().reshape(nb,k//32,512)
 raw,scale=module.unpack(packed,scales)
 codes=np.empty((n,k),np.uint8); codes[:,::2]=raw&15; codes[:,1::2]=raw>>4
 return lut[torch.from_numpy(codes).long()]*torch.exp2(torch.from_numpy(scale).float()-127).repeat_interleave(32,-1)

def quant(v,policy):
 v=v.float()
 if policy=='none':return v
 if policy=='native_e4_pow2':
  scale=torch.exp2(torch.ceil(torch.log2(v.abs().amax(-1,keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)/240)))
  q=(v/scale).to(torch.float8_e4m3fn).float()
  # Native Gaudi E4M3 finite subset, including subnormals.
  assert (q.abs()<=240).all()
  return q*scale
 if policy=='mxfp8':
  q,s=gpu_quant(v)
  return q.float()*torch.exp2(s.float()-127).repeat_interleave(32,-1)
 if policy=='mxfp8_native_rebase':
  # Preserve the GPU K32 quantizer first. Native E4M3 has max 240; convert
  # those already-quantized values to one row scale for a plain FP8 MME.
  # This can round small values: report it instead of calling it lossless.
  q,s=gpu_quant(v)
  gpu=q.float()*torch.exp2(s.float()-127).repeat_interleave(32,-1)
  scale=torch.exp2(torch.ceil(torch.log2(gpu.abs().amax(-1,keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)/240)))
  native=(gpu/scale).to(torch.float8_e4m3fn).float()
  assert (native.abs()<=240).all()
  restored=native*scale
  delta=restored-gpu
  native_rebase_stats.append(dict(elements=gpu.numel(),changed=int((delta!=0).sum()),
   exact_rows=int((delta==0).all(-1).sum()),rows=gpu.shape[0],
   error_sum_sq=float(delta.double().square().sum()),reference_sum_sq=float(gpu.double().square().sum())))
  return restored
 raise ValueError('unknown single-component policy: '+policy)

def hidden(gu):
 g,u=gu.to(torch.bfloat16).chunk(2,-1)
 return (torch.nn.functional.silu(g.float()).to(torch.bfloat16)*u).to(torch.bfloat16).float()

arms={'w4a16':('none','none'),'gpu_mxfp8_gp_only':('mxfp8','none'),
      'gpu_mxfp8_down_only':('none','mxfp8'),'gpu_mxfp8_both':('mxfp8','mxfp8'),
      'native_e4_pow2_both':('native_e4_pow2','native_e4_pow2'),
      'gpu_mxfp8_native_rebase_both':('mxfp8_native_rebase','mxfp8_native_rebase')}
outputs={name:torch.zeros(a.tokens,8,6144) for name in arms}
totals={name:{k:[0.,0.,0.] for k in ('activation','gp','hidden','down')} for name in arms}
def add(name,key,got,ref):
 delta=got.double()-ref.double(); z=totals[name][key]
 z[0]+=float(delta.square().sum()); z[1]+=float(ref.double().square().sum());z[2]=max(z[2],float(delta.abs().max()))

weight_ranges=[]
with torch.inference_mode():
 experts=ids.unique().tolist()
 for i,e in enumerate(experts):
  rr,ss=(ids==e).nonzero(as_tuple=True); xx=x[rr].float(); wg=weight(e,False); wd=weight(e,True)
  assert torch.isfinite(wg).all() and torch.isfinite(wd).all()
  for stage,w in (('gp',wg),('down',wd)):
   scale=torch.exp2(torch.ceil(torch.log2(w.abs().amax(-1,keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)/240)))
   native=(w/scale).to(torch.float8_e4m3fn).float()
   changed=(native*scale!=w)
   weight_rebase_stats.append(dict(expert=e,stage=stage,elements=w.numel(),changed=int(changed.sum()),
    columns=w.shape[0],inexact_columns=int(changed.any(-1).sum())))
  gu0=xx@wg.T; h0=hidden(gu0); y0=h0@wd.T
  # Weight range diagnostic only; not a completed E4M3 weight converter.
  for stage,w in (('gp',wg),('down',wd)):
   nz=w!=0; ex=torch.floor(torch.log2(w.abs().clamp_min(torch.finfo(torch.float32).tiny)))
   lo=torch.where(nz,ex,float('inf')).amin(-1);hi=torch.where(nz,ex,-float('inf')).amax(-1)
   weight_ranges.append(dict(expert=e,stage=stage,max_exponent_span=float((hi-lo).max()),all_columns_fit_e4m3_normal=bool(((hi-lo)<=13).all())))
  for name,(qp,qd) in arms.items():
   xxq=quant(xx,qp); gu=gu0 if qp=='none' else xxq@wg.T
   h=h0 if qp=='none' else hidden(gu); hq=quant(h,qd); y=y0 if name=='w4a16' else hq@wd.T
   outputs[name][rr,ss]=y
   for key,v,ref in [('activation',xxq,xx),('gp',gu,gu0),('hidden',h,h0),('down',y,y0)]: add(name,key,v,ref)
  if (i+1)%32==0:print(json.dumps(dict(experts_done=i+1,total=len(experts))),flush=True)
 # Same route-slot order, FP32 multiply/add in all arms. Not a claim of TPC FMA bit identity.
 combined={}
 for name,y in outputs.items():
  out=torch.zeros(a.tokens,6144)
  for slot in range(8):out=out+y[:,slot]*routes[:,slot,None]
  combined[name]=out

def metrics(v,r):
 v=v.double();r=r.double(); delta=v-r
 rel=torch.linalg.vector_norm(delta,dim=-1)/torch.linalg.vector_norm(r,dim=-1).clamp_min(1e-30)
 return dict(relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(r)),
             max_abs=float(delta.abs().max()),row_relative_l2_p50=float(rel.median()),
             row_relative_l2_p99=float(torch.quantile(rel,.99)),row_relative_l2_max=float(rel.max()),
             cosine=float((v.flatten()@r.flatten())/(v.norm()*r.norm())),finite=bool(torch.isfinite(v).all()))
result=dict(scope='CPU FP32 TP-local real checkpoint MoE, fixed actual routes; not full-model or hardware W4A8 qualification',
 model_quality_qualified=False,device_w4a8_qualified=False,tokens=a.tokens,active_experts=len(experts),
 fixture_sha256=sha(fixture),vllm_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.vllm,text=True).strip(),
 gpu_reference=dict(path=str(src),sha256=sha(src),function=node.name),
 intermediate_contract='FP32 matmul; BF16 gate/up; FP32 SiLU narrowed BF16; BF16 product; FP32 down and routing sum',
 metrics_vs_w4a16={name:metrics(v,combined['w4a16']) for name,v in combined.items()},
 metrics_vs_gpu_w4a8={name:metrics(v,combined['gpu_mxfp8_both']) for name,v in combined.items()},
 acceptance_reference='GPU single-component K32 MXFP8; W4A16 differences are quantization diagnostics, not implementation-error thresholds',
 stages={name:{key:dict(relative_l2=(z[0]/max(z[1],1e-30))**.5,max_abs=z[2]) for key,z in vals.items()} for name,vals in totals.items()},
 weight_range=dict(columns_fit_all=all(r['all_columns_fit_e4m3_normal'] for r in weight_ranges),
                   max_exponent_span=max(r['max_exponent_span'] for r in weight_ranges)),
 no_new_weight_quantization=True)
result['native_mxfp8_rebase']=dict(
 scope='GPU K32 MXFP8 values rebased to native E4M3 with one row scale; extra underflow/subnormal rounding explicitly measured',
 activation_totals={k:sum(r[k] for r in native_rebase_stats) for k in native_rebase_stats[0]},
 activation_relative_l2=(sum(r['error_sum_sq'] for r in native_rebase_stats)/max(1e-30,sum(r['reference_sum_sq'] for r in native_rebase_stats)))**.5,
 weight_exact_rebase_pass=all(r['changed']==0 for r in weight_rebase_stats),
 weight_changed_elements=sum(r['changed'] for r in weight_rebase_stats),
 weight_inexact_columns=sum(r['inexact_columns'] for r in weight_rebase_stats),
 note='If weight guard fails, this CPU exact-weight arm does not qualify a native FP8 weight implementation. No device/model claim.')
device_path=a.asset/'results/checkpoint-wide-all384-t512-d/checkpoint_routes-c32.pt'
assert sha(device_path)==manifest[str(device_path.relative_to(a.asset))]['sha256']
device=torch.load(device_path,weights_only=False,map_location='cpu')[0]['y'][rows]
result['cpu_reference_vs_archived_w4a16_device']=metrics(combined['w4a16'],device)
torch.save(dict(rows=rows,ids=ids,routing=routes,combined=combined,per_expert=outputs),a.out/'outputs.pt')
(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
(a.out/'weight-ranges.json').write_text(json.dumps(weight_ranges,indent=2)+'\n')
(a.out/'native-rebase-detail.json').write_text(json.dumps(dict(activation=native_rebase_stats,weights=weight_rebase_stats),indent=2)+'\n')
print(json.dumps(result,indent=2))
