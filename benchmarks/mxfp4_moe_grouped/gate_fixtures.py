"""Production GP shape with exactly summable FP32 MACs and literal gate oracle.

A in integer/128, scale codes124..127. All products are multiples of2^-11;
sumabs <= K*6/8, so at K6144 every partial's integer magnitude is <2^24.
The CPU FP64 dot is exact and its FP32 cast is exact. This removes MAC-order
ambiguity from the subsequent strict BF16 boundary comparison on the device.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,required=True);p.add_argument('--routes',type=int,default=2);p.add_argument('--hot',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'));from gaudi_kernels.mxfp4_prepared_gemv import prepare
out=a.output;out.mkdir(parents=True,exist_ok=False);N,K,E,T,R=512,6144,2,a.tokens,a.routes;assert 1<=R<=E and (not a.hot or R==1);rng=np.random.default_rng(28092026)
x=rng.integers(-16,17,(T,K)).astype(np.float32)/128;xb=(x.view(np.uint32)>>16).astype(np.uint16);assert np.array_equal((xb.astype(np.uint32)<<16).view(np.float32),x)
ids=(np.arange(T*R).reshape(T,R)%E).astype(np.int32)
if a.hot:ids[:]=0
routing=np.full((T,R),1/R,np.float32);lut=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6]);packed=[];scales=[];dots=[]
for e in range(E):
 rows=rng.integers(0,256,(N,K//2),dtype=np.uint8);s=rng.integers(124,128,(N,K//32),dtype=np.uint8);p=prepare(rows,s,K);packed.append(p.weight);scales.append(p.scales);q=np.empty((N,K),np.uint8);q[:,::2]=rows&15;q[:,1::2]=rows>>4;w=lut[q]*np.exp2(np.repeat(s.astype(np.int32)-127,32,axis=1));y=x.astype(np.float64)@w.T;assert np.array_equal(y,y.astype(np.float32).astype(np.float64));dots.append(y)
reference=np.stack([dots[int(ids[t,r])][t] for t in range(T) for r in range(R)]);order=np.concatenate([base+np.r_[np.arange(0,128,2),np.arange(1,128,2)] for base in range(0,N,128)])
reference[:,order].astype(np.float32).tofile(out/'reference_gate_partial.bin');reference.tofile(out/'gp_reference.f64');np.concatenate(packed).tofile(out/'packed.bin');np.concatenate(scales).tofile(out/'scales.bin');xb.tofile(out/'activation.bin');ids.tofile(out/'ids.bin');routing.tofile(out/'routing.bin');(out/'config.txt').write_text(f'{N} {K} {T} {R} {E}\n');counts=np.bincount(ids.reshape(-1),minlength=E)
(out/'layout.txt').write_text('native\n')
meta={'weight_layout':'native','N':N,'K':K,'T':T,'R':R,'E':E,'M_e':counts.tolist(),'active_experts':int(np.count_nonzero(counts)),'all_weight_scale_bytes':E*N*K*17//32,'active_weight_scale_bytes':int(np.count_nonzero(counts))*N*K*17//32,'mode':'gate','activation_per_token':True,'scale_range':[124,127],'fp32_mac_exact_lattice_exponent':-11,'sumabs_integer_upper':int(K*6/8*2**11),'gate_contract':'BF16 gate/up -> FP32 SiLU -> BF16 SiLU -> BF16 multiply','oracle':'full FP64 MAC exact here, then literal compact_gate.c on HPU','files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}};(out/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k!='files_sha256'}))
