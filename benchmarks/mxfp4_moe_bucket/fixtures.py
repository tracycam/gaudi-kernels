"""Aligned original-byte fixtures shared by bucket and capacity-T native probes.

Raw MACs use the exactly summable GP dyadic domain. Device gate mode compares
against literal TPC gate on full-FP64 raw dots, preserving SiLU implementation.
"""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,required=True);p.add_argument('--experts',type=int,default=2);p.add_argument('--routes',type=int,default=2);p.add_argument('--pattern',choices=['uniform','hot','ragged','boundary'],default='uniform');p.add_argument('--layout',choices=['native','historical'],default='native');p.add_argument('--mode',choices=['gate','linear'],default='gate');a=p.parse_args()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'));from gaudi_kernels.mxfp4_prepared_gemv import prepare
spec=importlib.util.spec_from_file_location('legacy_moe_pack',root/'csrc/tpc/mxfp4_moe/legacy/packing.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
N,K,E,T,R=512,6144,a.experts,a.tokens,a.routes
assert 1<=R<=E and 1<=T<=257 and E<=16, 'small native fixtures only; E384 is budgeted separately'
if a.pattern=='hot':assert R==1
ids=np.full((T,R),-1,np.int32)
if a.pattern=='boundary':
 counts=[16,17,32,33] if T==64 else [1,16,17,32,33,64,65,128,129,256,257]
 assert max(counts)<=T and sum(counts)<=T*R and len(counts)<=E
 at=0
 for e,count in enumerate(counts):
  for _ in range(count):ids[at%T,at//T]=e;at+=1
else:
 ids[:]=(np.arange(T*R).reshape(T,R)%E)
 if a.pattern=='hot':ids[:]=0
 if a.pattern=='ragged':ids[(np.arange(T)[:,None]+np.arange(R)[None,:])%3==0]=-1;ids[-1,-1]=E+3
assert all(len(set(row[(row>=0)&(row<E)]))==np.count_nonzero((row>=0)&(row<E)) for row in ids)
out=a.output;out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(29092026)
x=rng.integers(-16,17,(T,K)).astype(np.float32)/128;xb=(x.view(np.uint32)>>16).astype(np.uint16)
assert np.array_equal((xb.astype(np.uint32)<<16).view(np.float32),x)
routing=rng.integers(1,9,(T,R)).astype(np.float32)/16;lut=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6]);packed=[];scales=[];dots=[];norms=[]
for e in range(E):
 rows=rng.integers(0,256,(N,K//2),dtype=np.uint8);s=rng.integers(124,128,(N,K//32),dtype=np.uint8);weight=prepare(rows,s,K) if a.layout=='native' else None;w,scale=(weight.weight,weight.scales) if weight is not None else legacy.pack(rows,s);packed.append(w);scales.append(scale)
 q=np.empty((N,K),np.uint8);q[:,::2]=rows&15;q[:,1::2]=rows>>4;w=lut[q]*np.exp2(np.repeat(s.astype(np.int32)-127,32,axis=1));dot=x.astype(np.float64)@w.T
 assert np.array_equal(dot,dot.astype(np.float32).astype(np.float64));dots.append(dot);norms.append(np.abs(x.astype(np.float64))@np.abs(w).T)
raw=np.zeros((T*R,N),np.float64);normal=np.zeros_like(raw)
for t in range(T):
 for r in range(R):
  e=ids[t,r]
  if 0<=e<E:raw[t*R+r]=dots[e][t];normal[t*R+r]=norms[e][t]
order=np.concatenate([base+np.r_[np.arange(0,128,2),np.arange(1,128,2)] for base in range(0,N,128)])
raw[:,order].astype(np.float32).tofile(out/'reference_gate_partial.bin');raw.tofile(out/'gp_reference.f64')
(raw.reshape(T,R,N)*routing[:,:,None]).sum(axis=1).tofile(out/'reference.f64');(normal.reshape(T,R,N)*np.abs(routing[:,:,None])).sum(axis=1).tofile(out/'sumabs.f64')
np.concatenate(packed).tofile(out/'packed.bin');np.concatenate(scales).tofile(out/'scales.bin');(xb if a.mode=='gate' else np.repeat(xb,R,axis=0)).tofile(out/'activation.bin');ids.tofile(out/'ids.bin');routing.tofile(out/'routing.bin');(out/'config.txt').write_text(f'{N} {K} {T} {R} {E}\n')
(out/'layout.txt').write_text(a.layout+'\n')
counts=[int(np.count_nonzero(ids==e)) for e in range(E)]
meta={'N':N,'K':K,'E':E,'T':T,'R':R,'M_e':counts,'pattern':a.pattern,'mode':a.mode,'weight_layout':a.layout,'active_routes':sum(counts),'active_experts':sum(c>0 for c in counts),'all_weight_scale_bytes':E*N*K*17//32,'active_weight_scale_bytes':sum(c>0 for c in counts)*N*K*17//32,'scale_range':[124,127],'activation':'BF16 integer/128, magnitude<=1/8','fp32_mac_lattice_exponent':-11,'sumabs_integer_upper':9437184,'gate_contract':'BF16 gate/up -> FP32 SiLU -> BF16 SiLU -> BF16 multiply','oracle':'full FP64 original-byte MAC; literal TPC gate for GP','files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
(out/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k!='files_sha256'}))
