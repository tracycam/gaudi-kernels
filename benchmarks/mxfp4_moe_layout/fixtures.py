"""Byte-exact production packing and per-lane old-ASM/decoder audit inputs."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2];out=a.output;out.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('historical_pack',root/'csrc/tpc/mxfp4_moe/legacy/packing.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
sys.path.insert(0,str(root/'python'));from gaudi_kernels.mxfp4_prepared_gemv import prepare,unpack
summary=[];values=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float32)
for name,N,K,splits in [('gp',512,6144,3),('down',6144,256,1)]:
 case=out/name;case.mkdir();E=2;historical=[];native=[];hs=[];ns=[];x=np.zeros(K,np.float32);expected=[];packed_difference=0;scale_difference=0
 for part in range(splits):
  start=part*(K//splits)
  for j in range(7):x[start+j]=2**(2*j+1)
 for expert in range(E):
  n=np.arange(N)[:,None];k=np.arange(K)[None,:];q=((n+3*k+expert)%16).astype(np.uint8);s=(124+(n//8+np.arange(K//32)[None,:]+expert)%4).astype(np.uint8)
  for part in range(splits):
   start=part*(K//splits);labels=np.arange(N)+1+expert*N+part*(2*N);s[:,start//32]=127
   for j in range(7):q[:,start+j]=((labels>>(2*j))&3).astype(np.uint8)
  rows=q[:,::2]|(q[:,1::2]<<4);w,scale=legacy.pack(rows,s);prepared=prepare(rows,s,K);p2,s2=legacy.unpack(w,scale);assert np.array_equal(rows,p2) and np.array_equal(s,s2);p2,s2=unpack(prepared.weight,prepared.scales,N,K);assert np.array_equal(rows,p2) and np.array_equal(s,s2)
  packed_difference+=int(np.count_nonzero(w.reshape(-1)!=prepared.weight.reshape(-1)));scale_difference+=int(np.count_nonzero(scale.reshape(-1)!=prepared.scales.reshape(-1)));historical.append(w);hs.append(scale);native.append(prepared.weight);ns.append(prepared.scales)
  if expert==1:
   decoded=values[q]*np.exp2(np.repeat(s.astype(np.int32)-127,32,axis=1));bits=(decoded.astype(np.float32).view(np.uint32)>>16).astype(np.uint16);bits.T.tofile(case/'expected_decoded.bf16')
   for part in range(splits):
    lo=part*(K//splits);hi=(part+1)*(K//splits);dot=decoded[:,lo:hi].astype(np.float64)@x[lo:hi].astype(np.float64);labels=np.arange(N)+1+expert*N+part*(2*N);assert np.array_equal(dot,labels);expected.append(dot.astype(np.float32))
 np.concatenate(historical).tofile(case/'historical_weight.bin');np.concatenate(hs).tofile(case/'historical_scale.bin');np.concatenate(native).tofile(case/'native_weight.bin');np.concatenate(ns).tofile(case/'native_scale.bin');np.stack(expected).tofile(case/'expected_partial.f32');(x.view(np.uint32)>>16).astype(np.uint16).tofile(case/'activation.bf16');(case/'config.txt').write_text(f'{N} {K} {E} {splits}\n')
 meta={'shape':name,'N':N,'K':K,'E':E,'splits':splits,'all_original_weight_scale_bytes':E*N*K*17//32,'historical_vs_native_weight_bytes_differ':packed_difference,'historical_vs_native_scale_bytes_differ':scale_difference,'lossless_both':True,'selected_expert':1,'MAC_reference':'Each old-ASM split output is unique logical n+1+expert*N+split*(2*N), exactly representable FP32','files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in case.iterdir() if p.is_file()}}
 assert packed_difference and scale_difference;(case/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n');summary.append(meta)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps([{k:v for k,v in c.items() if k!='files_sha256'} for c in summary]))
