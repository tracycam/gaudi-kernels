"""Full-output actual ISA gates, never a device/acquire or MME test."""
import argparse,ctypes,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--old-gate',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
cmd=['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(root/'benchmarks/moe_route_tiles/isa_simulator.cpp'),'-L/usr/lib/habanatools','-Wl,-rpath,/usr/lib/habanatools','-ltpc_tests_core_ext','-ldl','-o',str(out/'simulator')]
(out/'compile-command.json').write_text(json.dumps(cmd)+'\n')
with(out/'compile.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=45)
records=[]
def run(name,guid,inputs,shape,dtype,params=(),old=False):
 d=out/name;d.mkdir();spec=[guid,str(len(inputs)),'1',str(len(params)),*map(str,params)]
 for i,x in enumerate(inputs):
  x=np.ascontiguousarray(x);path=d/f'input{i}.bin';x.tofile(path);typ={np.dtype('uint16'):'bf16',np.dtype('float32'):'f32',np.dtype('int32'):'i32'}[x.dtype]
  spec.extend([typ,str(x.ndim),*map(str,x.shape[::-1]),str(path)])
 typ={np.dtype('uint16'):'bf16',np.dtype('float32'):'f32'}[np.dtype(dtype)];path=d/'output.bin';spec.extend([typ,str(len(shape)),*map(str,shape[::-1]),str(path)]);(d/'spec.txt').write_text(' '.join(spec)+'\n')
 command=[str(out/'simulator'),str(a.library.resolve()),str(d/'spec.txt'),str(a.old_gate.resolve())if old else'-'];(d/'command.json').write_text(json.dumps(command)+'\n')
 with(d/'stdout.log').open('w')as f:subprocess.run(command,env={**os.environ,'TPC_RUNNER':'0'},stdout=f,stderr=subprocess.STDOUT,check=True,timeout=45)
 return np.fromfile(path,dtype=dtype).reshape(shape)
def check(name,got,want,**extra):
 mismatch=int(np.count_nonzero(got.view('u'+str(got.dtype.itemsize))!=want.view('u'+str(want.dtype.itemsize))))
 records.append(dict(case=name,words=got.size,mismatches=mismatch,**extra));assert mismatch==0,(name,mismatch)
status=np.zeros(1,dtype='i4')
try:
 # BF16 gather is a bit transport, including signed zero/subnormal/NaN payloads.
 for t,r,c,b,slot in [(1,1,1,1,0),(17,8,3,3,1),(513,8,16,2,2)]:
  x=(np.arange(t*6144,dtype='u4')%65536).astype('u2').reshape(t,6144)
  for changed in (0,1):
   mapping=((np.arange((slot+b+1)*c)*13+changed*7)%(t*r)).astype('i4');
   if changed:
    mapping[slot*c]=-1
    if b*c>1:mapping[slot*c+1]=t*r
   want=np.zeros((b,c,6144),dtype='u2')
   for local in range(b):
    for row in range(c):
     route=mapping[(slot+local)*c+row]
     if 0<=route<t*r:want[local,row]=x[route//r]
   name=f'gather-t{t}-changed{changed}';got=run(name,'gk_route_tile_gather_bf16_v1',[x,mapping,status],want.shape,'u2',(r,c,slot));check(name,got,want)
  got=run(f'gather-t{t}-bad','gk_route_tile_gather_bf16_v1',[x,mapping,np.array([1],dtype='i4')],want.shape,'u2',(r,c,slot));check(f'gather-t{t}-bad',got,np.zeros_like(want))
 # Compare the actual old qualified graph_gate_rows ELF, no exp/reciprocal approximation in oracle.
 rng=np.random.default_rng(912)
 for c,b,slot in [(1,1,0),(3,3,2),(16,2,1)]:
  partial=rng.uniform(-16,16,(b,c,512)).astype('f4');valid=np.zeros(slot+b+2,dtype='i4');valid[slot:slot+b]=[(i+1)% (c+1)for i in range(b)]
  # Include zero and BF16 half-way neighborhoods in valid rows.
  partial.flat[:8]=[0.,-0.,1.00390625,-1.00390625,20.,-20.,0.0078125,-0.0078125]
  for local in range(b):partial[local,valid[slot+local]:]=np.nan
  expert=np.arange(valid.size,dtype='i4').reshape(1,-1);counts=valid.reshape(1,-1)
  name=f'gate-c{c}-b{b}';got=run(name,'gk_route_tile_gate_bf16_v1',[partial,valid,status],(b,c,256),'u2',(slot,))
  old=run(name+'-old','gk_mxfp4_graph_gate_rows',[partial,expert,counts],(b,c,256),'u2',(slot,),True);check(name,got,old,padding_partial='NaN masked before load')
  for local in range(b):assert not got[local,valid[slot+local]:].any()
  got=run(name+'-bad','gk_route_tile_gate_bf16_v1',[np.full_like(partial,np.nan),valid,np.array([8],dtype='i4')],(b,c,256),'u2',(slot,));check(name+'-bad',got,np.zeros_like(old))
 # Independent stdlib correctly rounded fmaf, ascending original r, no GEMM oracle circularity.
 libm=ctypes.CDLL('libm.so.6');assert libm.fegetround()==0,'CPU reference requires FE_TONEAREST';fma=libm.fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
 for t,r,width in [(1,1,512),(3,2,512),(2,8,1024)]:
  rows=t*r+5;partial=rng.normal(size=(rows,width)).astype('f4');routing=rng.uniform(-1,1,(t,r)).astype('f4');inv=rng.permutation(rows)[:t*r].astype('i4');partial[[i for i in range(rows)if i not in inv]]=np.nan
  if r==8:
   routing[0]=1;partial[inv[:8],0]=[2**24,1,-2**24,1,0,0,0,0] # ordered=1, a different legal order can be2.
  want=np.zeros((t,width),dtype='f4')
  for token in range(t):
   for n in range(width):
    value=0.
    for route in range(r):value=fma(float(partial[inv[token*r+route],n]),float(routing[token,route]),value)
    want[token,n]=value
  name=f'combine-t{t}-r{r}';got=run(name,'gk_route_tile_combine_f32_v1',[partial,routing,inv,status],want.shape,'f4');check(name,got,want)
  for kind in ('status','negative','high'):
   st=status.copy();ii=inv.copy()
   if kind=='status':st[0]=1
   else:ii[0]=-1 if kind=='negative'else rows
   y=run(name+'-'+kind,'gk_route_tile_combine_f32_v1',[partial,routing,ii,st],want.shape,'f4');badrows=t if kind=='status'else 1
   assert np.isnan(y[:badrows]).all()
   check(name+'-'+kind,y[badrows:],want[badrows:],nan_words=int(badrows*width))
 report=dict(status='PASS_ACTUAL_ISA_ROUTE_TILE_CONSUMERS',device_qualified=False,MME_tested=False,records=records,CPU_FMA_rounding='FE_TONEAREST',total_words=sum(x['words']+x.get('nan_words',0)for x in records))
except Exception as exc:
 report=dict(status='FAIL',error=repr(exc),records=records,device_qualified=False);raise
finally:
 report['input_libraries']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()for p in (a.library,a.old_gate,Path('/usr/lib/habanatools/libtpc_tests_core_ext.so'))};(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ('records','input_libraries')}))
