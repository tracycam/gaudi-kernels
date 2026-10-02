"""Independent original-row FP64 projections + actual literal TPC gate ISA."""
import argparse,importlib.util,json,os,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--tokens',type=int,default=2);p.add_argument('--case',choices=['mixed','hot','ragged'],default='mixed');a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('historical_packing',root/'csrc/tpc/mxfp4_moe/legacy/packing.py');packing=importlib.util.module_from_spec(spec);spec.loader.exec_module(packing)
E,T,R=2,a.tokens,2;rng=np.random.default_rng(873)
values=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float64)
x=(rng.integers(-16,17,(T,6144))/1024).astype(np.float32);ids=np.tile(np.arange(R,dtype=np.int32),(T,1));route=(rng.integers(-4,5,(T,R))/8).astype(np.float32)
if a.case=='hot':ids[:,1]=-1
if a.case=='ragged':ids[::2,1]=-1;ids[-1,0]=E+3
owners={};dense={}
for name,N,K in [('gp',512,6144),('down',6144,256)]:
 ws,ss,exacts=[],[],[]
 for e in range(E):
  q=rng.integers(0,16,(N,K),dtype=np.uint8);sc=rng.integers(118,121,(N,K//32),dtype=np.uint8)
  rows=q[:,::2]|(q[:,1::2]<<4);w,s=packing.pack(rows,sc);r2,s2=packing.unpack(w,s);assert np.array_equal(rows,r2) and np.array_equal(sc,s2)
  np.save(out/f'{name}-e{e}-original.npy',rows);np.save(out/f'{name}-e{e}-original-scales.npy',sc)
  ws.append(w.reshape(-1,256));ss.append(s.reshape(-1,512));exacts.append(values[q]*np.exp2(np.repeat(sc.astype(np.int32)-127,32,axis=1)))
 owners[name]=np.concatenate(ws);owners[name+'s']=np.concatenate(ss);dense[name]=exacts
raw=np.zeros((T,R,512),np.float64)
for t in range(T):
 for r in range(R):
  e=ids[t,r]
  if 0<=e<E:raw[t,r]=x[t].astype(np.float64)@dense['gp'][e].T
assert np.array_equal(raw,raw.astype(np.float32).astype(np.float64))
raw.astype(np.float32).tofile(out/'gp-mac-f32.bin')
with (out/'literal-reference.log').open('w') as log:subprocess.run([str(a.reference.resolve()),str(T*R),str(out/'gp-mac-f32.bin'),str(out/'gate-bf16.bin')],env={**os.environ,'TPC_RUNNER':'0'},stdout=log,stderr=subprocess.STDOUT,timeout=45,check=True)
gate=(np.fromfile(out/'gate-bf16.bin',np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(T,R,256)
y=np.zeros((T,6144),np.float32);absolute=np.zeros_like(y,dtype=np.float64);rawdown=np.zeros((T,R,6144),np.float64)
for t in range(T):
 for r in range(R):
  e=ids[t,r]
  if 0<=e<E:
   partial=gate[t,r].astype(np.float64)@dense['down'][e].T;rawdown[t,r]=partial
   # Match old ordered FP32 FMA combine after each FP32 down result.
   y[t]=(partial.astype(np.float32).astype(np.float64)*float(route[t,r])+y[t].astype(np.float64)).astype(np.float32)
   absolute[t]+=(np.abs(gate[t,r].astype(np.float64))@np.abs(dense['down'][e]).T)*abs(float(route[t,r]))
for name,data in {**owners,'x':x,'ids':ids,'routing':route,'expected':y,'absolute':absolute,'gp_exact':raw,'gate_bits':(gate.view(np.uint32)>>16).astype(np.uint16),'down_exact':rawdown}.items():np.save(out/(name+'.npy'),data)
lut=np.stack([values[np.arange(256)&15],values[np.arange(256)>>4]],axis=1).astype(np.float32).reshape(1,512);np.save(out/'lut.npy',lut)
(out/'fixture.json').write_text(json.dumps(dict(E=E,T=T,R=R,case=a.case,layout='HistoricalN512',finite_scale_range=[118,120],activation='integer[-16,16]/1024',GP_full_FP64_exactly_representable_in_FP32=True,gate_oracle='actual literal compact_gate TPC ISA',down_oracle='FP64 original nibble/scales, rounded FP32, then ordered FP32 FMA routing',active_experts=len(set(int(v) for v in ids.flat if 0<=v<E)),device_validated=False),indent=2)+'\n')
print(out)
