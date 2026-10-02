import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,required=True);p.add_argument('--n',type=int,default=6144);p.add_argument('--k',type=int,default=256);p.add_argument('--experts',type=int,default=16);p.add_argument('--routes',type=int,default=8);p.add_argument('--distribution',choices=['uniform','skew','ragged'],default='uniform');a=p.parse_args();root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'));from gaudi_kernels.mxfp4_prepared_gemv import prepare
import importlib.util
spec=importlib.util.spec_from_file_location('legacy_packing',root/'reference/mxfp4/executor/kernels/packing.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
out=a.output;out.mkdir(parents=True,exist_ok=False);n,k,t,e,r=a.n,a.k,a.tokens,a.experts,a.routes;rng=np.random.default_rng(28092026);q=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],np.float64)
raw_a=rng.integers(-256,257,(t*r,k)).astype(np.float32)/256;u=raw_a.view(np.uint32);ab=((u+32767+((u>>16)&1))>>16).astype(np.uint16);aa=(ab.astype(np.uint32)<<16).view(np.float32)
ids=(np.arange(t*r)%e).astype(np.int32).reshape(t,r)
if a.distribution=='skew':ids[:]=0
if a.distribution=='ragged':ids[(np.arange(t)[:,None]+np.arange(r)[None,:])%5==0]=-1
routes=rng.uniform(.031,.22,(t,r)).astype(np.float32);routes/=routes.sum(1,keepdims=True);routes[ids<0]=0
reference=np.zeros((t,n),np.float64);absolute=np.zeros_like(reference);packed=[];exponents=[];legacy_packed=[];legacy_scales=[];occupancy=[]
for expert in range(e):
 rows=rng.integers(0,256,(n,k//2),dtype=np.uint8);sc=rng.integers(118,133,(n,k//32),dtype=np.uint8);prepared=prepare(rows,sc,k);packed.append(prepared.weight);exponents.append(prepared.scales);oldw,olds=legacy.pack(rows,sc);legacy_packed.append(oldw);legacy_scales.append(olds)
 selected=np.flatnonzero(ids.reshape(-1)==expert);occupancy.append(len(selected))
 if len(selected):
  for start in range(0,n,256):
   end=min(n,start+256);code=np.empty((end-start,k),np.uint8);code[:,::2]=rows[start:end]&15;code[:,1::2]=rows[start:end]>>4
   w=q[code]*np.exp2(np.repeat(sc[start:end].astype(np.int32)-127,32,axis=1));partial=aa[selected].astype(np.float64)@w.T;norm=np.abs(aa[selected].astype(np.float64))@np.abs(w).T
   for i,route in enumerate(selected):reference[route//r,start:end]+=partial[i]*float(routes.reshape(-1)[route]);absolute[route//r,start:end]+=norm[i]*abs(float(routes.reshape(-1)[route]))
np.concatenate(legacy_packed).tofile(out/'legacy_packed.bin');np.concatenate(legacy_scales).tofile(out/'legacy_scales.bin');np.concatenate(packed).tofile(out/'packed.bin');np.concatenate(exponents).tofile(out/'scales.bin');ab.tofile(out/'activation.bin');ids.tofile(out/'ids.bin');routes.tofile(out/'routing.bin');reference.tofile(out/'reference.f64');absolute.tofile(out/'sumabs.f64');(out/'config.txt').write_text(f'{n} {k} {t} {r} {e}\n')
meta={'N':n,'K':k,'T':t,'R':r,'E':e,'distribution':a.distribution,'M_e':occupancy,'active_routes':int(sum(occupancy)),'weight_layout':'native-v2','scale_range':[118,132],'activation':'unchanged BF16 dyadic <=1','routing':'FP32 normalized host fixture; device consumes without narrowing','oracle':'full FP64 MAC and routing products','files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}};(out/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k!='files_sha256'}))
