"""Full-output CPU gates and reduction-only fixtures; no device operations."""
import argparse,ctypes,json,hashlib
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--cpu-library',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--measured',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(270928)
fn=ctypes.CDLL(str(a.cpu_library.resolve())).reduce_cpu;fn.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*4
records=[]
def emit(name,part,sa,ws,bias,reference,scope,shape):
 g,m,n=part.shape;d=a.out/name;d.mkdir();arrays=[np.ascontiguousarray(v,dtype=np.float32) for v in [part,sa,ws,bias]]
 for k,v in zip(['partial','activation-scales','weight-scales','bias'],arrays):v.tofile(d/(k+'.bin'))
 reference=np.asarray(reference,dtype=np.float64);term_bound=(np.abs(arrays[0].astype(np.float64))*arrays[1].astype(np.float64)[:,:,None]*arrays[2].repeat(128,axis=0)[:n].T.astype(np.float64)[:,None,:]).sum(0)+np.abs(arrays[3].astype(np.float64));term_bound.tofile(d/'absolute-terms-fp64.bin');reference.tofile(d/'reference-fp64.bin');rows=[]
 for chains in [1,4,8,0]:
  out=np.empty((m,n),np.float32);bf=np.empty((m,n),np.uint16);fn(*[v.ctypes.data for v in arrays+[out,bf]],g,m,n,chains)
  out.tofile(d/f'cpu-chain{chains}-f32.bin');bf.tofile(d/f'cpu-chain{chains}-bf16.bin')
  y=torch.from_numpy(bf.view(np.int16)).view(torch.bfloat16).double().numpy();delta=y-reference;fdelta=out.astype(np.float64)-reference
  backward=np.divide(np.abs(delta),term_bound,out=np.zeros_like(delta),where=term_bound!=0);fbackward=np.divide(np.abs(fdelta),term_bound,out=np.zeros_like(fdelta),where=term_bound!=0);backward[(term_bound==0)&(delta!=0)]=np.inf;fbackward[(term_bound==0)&(fdelta!=0)]=np.inf
  rel=float(np.linalg.norm(delta)/max(np.linalg.norm(reference),1e-30));finite=bool(np.isfinite(y).all());rows.append(dict(chains=chains,mode='neumaier_product_residual' if chains==0 else 'chains'+str(chains),finite=finite,max_backward_error=float(backward.max()),fp32_max_backward_error=float(fbackward.max()),max_abs=float(np.max(np.abs(delta))),relative_l2=rel,fp32_max_abs=float(np.max(np.abs(fdelta))),fp32_relative_l2=float(np.linalg.norm(fdelta)/max(np.linalg.norm(reference),1e-30)),bf16_mismatches=int(np.count_nonzero(bf!=torch.from_numpy(reference).bfloat16().view(torch.int16).numpy().view(np.uint16))),existing_adapted_fp64_gate_pass=finite and rel<.006))
 record=dict(name=name,shape=shape,partial_shape=[g,m,n],scope=scope,rows=rows);(d/'case.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
# Actual MME partials at full production QKV dimensions, full original FP64 MAC.
part=np.fromfile(a.measured/'eager-partial-f32.bin',np.float32).reshape(48,1,3392);sa=np.fromfile(a.measured/'eager-activation-scales-f32.bin',np.float32).reshape(48,1);ws=np.fromfile(a.checkpoint/'scales-f32.bin',np.float32).reshape(27,48)*2;ref=np.fromfile(a.checkpoint/'qkv-full-fp64-reference.bin',np.float64).reshape(1,3392)
emit('qkv-measured-m1',part,sa,ws,np.zeros(3392,np.float32),ref,'measured device MME partials; full original adapted FP64 MAC',[1,3392,6144])
# Original framework CPU/tail shapes plus bounded M coverage and G tails.
shapes=[(1,129,129),(3,257,513),(16,128,128),(2,129,129),(8,127,385),(32,129,769),(64,257,1025),(128,129,257),(256,1,513),(257,129,257),(512,129,257),(513,1,33),(16,3392,6144)]
for index,(m,n,k) in enumerate(shapes):
 g=(k+127)//128;kind='zero' if index==3 else 'cancellation' if index==4 else 'wide-range' if index==6 else 'random'
 x=torch.randn(m,k).bfloat16();raw=(torch.randn(n,k)*64).clamp(-448,448).to(torch.float8_e4m3fn);scales=torch.rand((n+127)//128,g)*.0002+.0001;bias=torch.randn(n)*.01
 if (m,n,k)==(16,3392,6144):raw=torch.from_numpy(np.fromfile(a.checkpoint/'weight-e4m3.bin',np.uint8).copy()).view(torch.float8_e4m3fn).reshape(n,k);scales=torch.from_numpy(np.fromfile(a.checkpoint/'scales-f32.bin',np.float32).copy()).reshape(27,48)
 if kind=='zero':x.zero_();x[:,1::2]=-0.0
 if kind=='cancellation':
  x.fill_(1);x[:,1::2]=-1;raw.view(torch.uint8)[:,1:(k//2)*2:2]=raw.view(torch.uint8)[:,:k//2*2:2]
  if k%2:raw.view(torch.uint8)[:,-1]=0
 if kind=='wide-range':scales*=torch.pow(2.,torch.randint(-40,41,scales.shape));x=(x.float()*torch.pow(2.,torch.randint(-30,31,(m,1)))).bfloat16()
 padded=torch.zeros(m,g*128);padded[:,:k]=x.float();blocks=padded.reshape(m,g,128).permute(1,0,2).contiguous();s=blocks.abs().amax(-1,keepdim=True).clamp_min(1e-10)*(1./448.);qa=((blocks/s).clamp(-448,448).to(torch.float8_e4m3fn).float()*.5).to(torch.float8_e4m3fn).double();s=s[:,:,0]*2
 paddedw=torch.zeros(n,g*128);paddedw[:,:k]=(raw.float()*.5).to(torch.float8_e4m3fn).float();w=paddedw.reshape(n,g,128).permute(1,2,0).double();partials=torch.bmm(qa,w);sw=scales*2
 reference=(partials*s.double()[:,:,None]*sw.repeat_interleave(128,0)[:n].T.double()[:,None,:]).sum(0)+bias.double()
 emit(f'{kind}-m{m}-n{n}-k{k}',partials.float().numpy(),s.numpy(),sw.numpy(),bias.numpy(),reference.numpy(),'synthetic once-rounded FP64 block dots; not a claim about actual MME tree',[m,n,k])
# Not hidden by an absolute tolerance: changing legal FP32 summation order can
# regress a cancellation-sensitive input that the original order computes well.
g,m,n=17,1,129;part=np.zeros((g,m,n),np.float32);part[0]=2**30;part[1]=-2**30;part[2]=1;part[8]=1
emit('order-sensitive-large-range',part,np.ones((g,m),np.float32),np.ones((2,g),np.float32),np.zeros(n,np.float32),np.full((m,n),2.,np.float64),'explicit reduction order counterexample; factors and products finite normal',[m,n,g*128])
report=dict(device_tested=False,gate={'adapted_fp64_relative_l2_lt':.006,'source':'existing block_fp8_framework/device.py; not changed'},all_accuracy_gates_pass=all(r['existing_adapted_fp64_gate_pass'] for c in records for r in c['rows']),records=records)
(a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(cases=len(records),all_accuracy_gates_pass=report['all_accuracy_gates_pass'],failed=[dict(name=c['name'],chains=r['chains'],relative_l2=r['relative_l2']) for c in records for r in c['rows'] if not r['existing_adapted_fp64_gate_pass']]),indent=2))
