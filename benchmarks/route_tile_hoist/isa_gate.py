"""Actual old/new decoder and gather ELFs, complete storage-bit CPU oracles."""
import argparse,hashlib,importlib.util,json,os,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--builds',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--random-codes',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
cmd=['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(root/'benchmarks/route_tile_hoist/isa_runner.cpp'),'-L/usr/lib/habanatools','-Wl,-rpath,/usr/lib/habanatools','-ltpc_tests_core_ext','-ldl','-o',str(out/'runner')]
(out/'compile-command.json').write_text(json.dumps(cmd)+'\n')
with(out/'compile.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=45)
spec=importlib.util.spec_from_file_location('packing',root/'csrc/tpc/mxfp4_moe/legacy/packing.py');packing=importlib.util.module_from_spec(spec);spec.loader.exec_module(packing)
records=[]
def run(name,stage,variant,inputs,shape,params):
 d=out/(name+'-'+variant);d.mkdir();guid='gk_mxfp4_graph_decode_historical'if stage=='core'else'gk_route_tile_gather_bf16_v1';s=[guid,str(len(inputs)),'1',str(len(params)),*map(str,params)]
 for i,x in enumerate(inputs):
  x=np.ascontiguousarray(x);f=d/f'input{i}.bin';x.tofile(f);typ={np.dtype('uint8'):'u8',np.dtype('uint16'):'bf16',np.dtype('int32'):'i32'}[x.dtype];s.extend([typ,str(x.ndim),*map(str,x.shape[::-1]),str(f)])
 path=d/'actual.bin';s.extend(['bf16',str(len(shape)),*map(str,shape[::-1]),str(path)]);(d/'spec.txt').write_text(' '.join(s)+'\n')
 library=a.builds/(stage+'-'+variant)/('libmxfp4_moe_graph_tpc.so'if stage=='core'else'libgaudi_route_tiles_tpc.so');command=[str(out/'runner'),str(library.resolve()),str(d/'spec.txt'),'-'];(d/'command.json').write_text(json.dumps(command)+'\n')
 with(d/'sim.log').open('w')as f:subprocess.run(command,env={**os.environ,'TPC_RUNNER':'0'},stdout=f,stderr=subprocess.STDOUT,check=True,timeout=45)
 return np.fromfile(path,dtype='u2').reshape(shape)
def gate(name,stage,inputs,want,params):
 old=run(name,stage,'control',inputs,want.shape,params);new=run(name,stage,'hoist',inputs,want.shape,params);want.tofile(out/(name+'-cpu.bin'))
 row=dict(case=name,stage=stage,words=want.size,old_CPU_mismatches=int((old!=want).sum()),new_CPU_mismatches=int((new!=want).sum()),old_new_mismatches=int((old!=new).sum()));records.append(row);(out/'partial-result.json').write_text(json.dumps(records,indent=2)+'\n');assert not any(row[k]for k in row if k.endswith('mismatches')),row
try:
 rng=np.random.default_rng(20928)
 values=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float32);bits=(values.view('u4')>>16).astype('u2');lut=bits[np.array([[i&15,i>>4]for i in range(256)])].reshape(1,512)
 for k,n,nt,start,slot,mapping in [(64,1024,1024,0,1,[3,1,-1,2,0]),(256,6144,2048,4,0,[2,-1,1,3]),(6144,512,512,0,1,[3,0,-1,2])]:
  e=4;w=[];sc=[];original=[]
  for expert in range(e):
   codes=((np.arange(n)[:,None]+3*np.arange(k)[None,:]+7*expert)%16).astype('u1');scales=(2+(np.arange(n)[:,None]*17+np.arange(k//32)[None,:]*29+expert*11)%251).astype('u1')
   if a.random_codes:codes=rng.integers(0,16,(n,k),dtype=np.uint8);scales=rng.integers(2,253,(n,k//32),dtype=np.uint8)
   packed=codes[:,::2]|(codes[:,1::2]<<4);wp,sp=packing.pack(packed,scales);unpacked,us=packing.unpack(wp,sp);assert np.array_equal(packed,unpacked)and np.array_equal(scales,us);w.append(wp);sc.append(sp)
   q=bits[codes].astype('u4');mag=q&32767;exp=np.repeat(scales.astype('i4'),32,axis=1);adjust=(mag.astype('i4')+((exp-127)<<7))&32767;adjust[mag==0]=0;original.append(((q&32768)|adjust).astype('u2'))
  wp=np.stack(w).reshape(-1,256);sp=np.stack(sc).reshape(-1,512);mp=np.array(mapping,dtype='i4').reshape(1,-1);batch=len(mapping)-slot;want=np.zeros((batch,k,nt),dtype='u2')
  for i,expert in enumerate(mapping[slot:]):
   if expert>=0:want[i]=original[expert][start*512:start*512+nt].T
  gate(f'decode-k{k}-nt{nt}','core',[wp,sp,lut,mp],want,(n//512,slot,start))
 for r in range(1,9):
  t,c,slot,batch=17,3,1,3;x=(np.arange(t*6144)%65536).astype('u2').reshape(t,6144);mapping=((np.arange((slot+batch+1)*c)*11+3)%(t*r)).astype('i4');mapping[slot*c]=-1;mapping[slot*c+1]=t*r;want=np.zeros((batch,c,6144),dtype='u2')
  for tile in range(batch):
   for row in range(c):
    route=mapping[(slot+tile)*c+row]
    if 0<=route<t*r:want[tile,row]=x[route//r]
  gate(f'gather-r{r}','tiles',[x,mapping,np.zeros(1,dtype='i4')],want,(r,c,slot))
 gate('gather-bad-status','tiles',[x,mapping,np.ones(1,dtype='i4')],np.zeros_like(want),(r,c,slot))
 report=dict(status='PASS_ACTUAL_ISA_HOIST',records=records,total_words=sum(r['words']for r in records),device_tested=False,MME_tested=False,fixture='nonperiodic_random_20928'if a.random_codes else'coordinate_pattern')
except Exception as exc:report=dict(status='FAIL',error=repr(exc),records=records);raise
finally:
 report['libraries']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in a.builds.glob('*/*.so')};(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ('libraries','records')}))
