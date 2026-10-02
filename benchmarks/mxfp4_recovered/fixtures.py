"""Deterministic original-width fixtures, prepared on CPU before HPU acquisition."""
import argparse, hashlib, importlib.util, json, sys
from fractions import Fraction
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'python'),str(root/'benchmarks/mxfp4_exact')]
from gaudi_kernels.mxfp4_compact import prepare
from verify import bf16, pow2, round_fraction, Q, f32_fraction
from guard import safe

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pattern',choices=['random','subnormal','mixed','cancellation','ties','finite-scales','copy'],required=True)
p.add_argument('--n',type=int,required=True);p.add_argument('--k',type=int,required=True);p.add_argument('--m',type=int,default=1);p.add_argument('--bias',action='store_true');a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
n,k,m=a.n,a.k,a.m;rng=np.random.default_rng(27092026)
def tobf(x):
 u=np.asarray(x,np.float32).view(np.uint32);return ((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)
def frombf(x):return (np.asarray(x,np.uint32)<<16).view(np.float32)
x=tobf(rng.integers(-1024,1025,(m,k)).astype(np.float32)/1024)
bias=(rng.integers(-50,51,n).astype(np.float32)/256).view(np.uint32) if a.bias else np.zeros(n,np.uint32)
if a.pattern=='copy':
 x.tofile(out/'activation.bf16');(out/'config.txt').write_text(f'{n} {k} {m} 0 118 132\n')
 meta={'N':n,'K':k,'M':m,'pattern':a.pattern,'weight_fixture':False}
else:
 rows=rng.integers(0,256,(n,(k+1)//2),dtype=np.uint8);scales=rng.integers(118,133,(n,(k+31)//32),dtype=np.uint8)
 if a.pattern!='random':
  codes=np.zeros((n,k),np.uint8)
  if a.pattern in ['subnormal','mixed']:
   x[:]=1 if a.pattern=='subnormal' else 0
   if a.pattern=='mixed':
    z=np.arange(k);x[:]=(((z%127+1)|np.where(z%2,0x8000,0)).astype(np.uint16))[None,:]
   codes[:]=(np.arange(n,dtype=np.uint32)[:,None]+(2 if a.pattern=="subnormal" else 0))%16
   scales[:]=127 if a.pattern=='subnormal' else np.array([2,9,10,127,252],np.uint8)[(np.arange(n)//16)%5,None]
  elif a.pattern=='cancellation':
   assert k>=3;x[:]=0;x[:,0]=x[:,2]=0x7f7f;x[:,1]=1;codes[:,0]=7;codes[:,1]=2;codes[:,2]=15;scales[:]=252
  elif a.pattern=='ties':
   assert k>=33;x[:]=0;x[:,0]=x[:,32]=1;scales[:]=0;scales[:,0]=111
   codes[:,0]=np.where(np.arange(n)%2,9,1);codes[:,32]=np.where((np.arange(n)//2)%2,codes[:,0],0)
   if a.bias:bias[:]=0;bias[4::8]=0x007fffff
  elif a.pattern=='finite-scales':
   assert k==1 and n==4080 and m==3
   x[0,:]=1;x[1,:]=0x3f80;x[2,:]=0x7f7f;codes[:,0]=np.arange(n)%16;scales[:,0]=np.arange(n)//16
  rows[:]=0;rows[:,:]=codes[:,::2]
  rows[:,:k//2]|=codes[:,1::2]<<4
  del codes
 prepared=prepare(rows,scales,logical_k=k)
 prepared.weight.tofile(out/'packed.u8');prepared.scales.tofile(out/'scales.u8');x.tofile(out/'activation.bf16')
 if a.bias:bias.tofile(out/'bias.u32')
 refs=np.empty((m,n),np.float64);absolute=np.empty((m,n),np.float64);expected=np.empty((m,n),np.uint32)
 if a.pattern=='random':
  for lo in range(0,n,128):
   hi=min(n,lo+128);codes=np.empty((hi-lo,rows.shape[1]*2),np.uint8);codes[:,::2]=rows[lo:hi]&15;codes[:,1::2]=rows[lo:hi]>>4
   w=np.array([float(q) for q in Q])[codes[:,:k]]*np.exp2(np.repeat(scales[lo:hi].astype(np.int32)-127,32,axis=1)[:,:k])
   xx=frombf(x).astype(np.float64);bb=bias[lo:hi].view(np.float32).astype(np.float64)
   refs[:,lo:hi]=xx@w.T+bb[None,:];absolute[:,lo:hi]=np.abs(xx)@np.abs(w).T+np.abs(bb)[None,:]
  expected[:]=refs.astype(np.float32).view(np.uint32)
 else:
  for col in range(n):
   weights=[Q[(int(rows[col,z//2])>>(4*(z%2)))&15]*pow2(int(scales[col,z//32])-127) for z in range(k)]
   for row in range(m):
    terms=[bf16(int(x[row,z]))*weights[z] for z in range(k)];b=f32_fraction(int(bias[col]))
    total=sum(terms,b);refs[row,col]=float(total);absolute[row,col]=float(sum(map(abs,terms),abs(b)));expected[row,col]=round_fraction(total)
 refs.tofile(out/'reference.f64');absolute.tofile(out/'sumabs.f64');expected.tofile(out/'expected.u32')
 smin,smax=int(scales.min()),int(scales.max());flags=np.ones((m,(n+511)//512),np.uint16)
 for row in range(m):
  for block in range((n+511)//512):flags[row,block]=not safe(x[row],smin,smax,n,k,bias[block*512:min(n,(block+1)*512)].tolist())
 flags.tofile(out/'expected_flags.u16')
 (out/'config.txt').write_text(f'{n} {k} {m} {int(a.bias)} {smin} {smax}\n')
 meta={'N':n,'K':k,'M':m,'pattern':a.pattern,'bias':a.bias,'layout_version':3,'packed_bytes':prepared.weight.nbytes,'scale_bytes':prepared.scales.nbytes,'scale_min':smin,'scale_max':smax,'guard_flags':flags.tolist(),'oracle':'FP64 exact dyadic sum' if a.pattern=='random' else 'independent Fraction sum and FP32 nearest-even distance','weight_fixture':True}
meta['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()};(out/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k!='files_sha256'}))
