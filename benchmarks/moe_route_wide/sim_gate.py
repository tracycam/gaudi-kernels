"""Actual unchanged ELF execution at C64/C128, TPC_RUNNER=0 only."""
import argparse,ctypes,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'benchmarks/moe_route_metadata_v3'))
from reference import reference
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--core-library',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build=a.build.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
pins=json.loads((root/'benchmarks/route_bundle/qualified_row_inputs.json').read_text());assert hashlib.sha256(a.core_library.read_bytes()).hexdigest()==pins['core']['library_sha256']
h=ctypes.CDLL(str(a.core_library.resolve()),mode=ctypes.RTLD_LOCAL);begin=ctypes.addressof(ctypes.c_ubyte.in_dll(h,'_binary_graph_gate_rows_o_start'));end=ctypes.addressof(ctypes.c_ubyte.in_dll(h,'_binary_graph_gate_rows_o_end'));blob=ctypes.string_at(begin,end-begin);assert hashlib.sha256(blob).hexdigest()==pins['core']['elfs']['graph_gate_rows']['elf_sha256'];old_gate=out/'old-gate.o';old_gate.write_bytes(blob)
lib=build/'libgaudi_route_wide_tpc.so';records=[]
def save(status='RUNNING',**kw):(out/'result.json').write_text(json.dumps(dict(status=status,device_qualified=False,MME_tested=False,records=records,library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),**kw),indent=2)+'\n')
def execute(command,d):
 (d/'command.json').write_text(json.dumps(command)+'\n')
 with(d/'stdout.log').open('w')as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'TPC_RUNNER':'0'},timeout=45,check=True)
def consumer(name,guid,inputs,shape,dtype,params=(),old=False):
 d=out/name;d.mkdir();spec=[guid,str(len(inputs)),'1',str(len(params)),*map(str,params)]
 types={np.dtype('uint16'):'bf16',np.dtype('float32'):'f32',np.dtype('int32'):'i32'}
 for i,x in enumerate(inputs):
  x=np.ascontiguousarray(x);path=d/f'input{i}.bin';x.tofile(path);spec.extend([types[x.dtype],str(x.ndim),*map(str,x.shape[::-1]),str(path)])
 target=d/'output.bin';spec.extend([types[np.dtype(dtype)],str(len(shape)),*map(str,shape[::-1]),str(target)]);(d/'spec.txt').write_text(' '.join(spec)+'\n')
 execute([str(build/'consumer_simulator'),str(lib),str(d/'spec.txt'),str(old_gate)if old else'-'],d)
 return np.fromfile(target,dtype=dtype).reshape(shape)
def check(name,got,want,**kw):
 bad=int(np.count_nonzero(got.view('u'+str(got.dtype.itemsize))!=want.view('u'+str(want.dtype.itemsize))))
 records.append(dict(case=name,words=got.size,mismatches=bad,**kw));save();assert not bad,(name,bad)
status=np.zeros(1,dtype='i4');rng=np.random.default_rng(20929)
try:
 # Metadata: changed routing, nonintegral final tiles, all fields complete-write.
 for c in(64,128):
  for kind in('uniform','hot','skew','duplicate','badid','overflow'):
   t,r,e=513,8,384;ids=np.asarray([[(i*8+j)%384 for j in range(r)]for i in range(t)],dtype='i4')
   if kind in('hot','skew'):ids[:t if kind=='hot'else 384]=np.arange(376,384,dtype='i4')
   if kind=='duplicate':ids[7,1]=ids[7,0]
   if kind=='badid':ids[9,2]=384
   b=384+(t*r-384)//c
   if kind=='overflow':b=1
   d=out/f'metadata-c{c}-{kind}';d.mkdir();ids.tofile(d/'ids.bin')
   command=[str(build/'metadata_simulator'),str(lib),str(d/'ids.bin'),str(d),str(t),str(r),str(e),str(c),str(b)];execute(command,d)
   want=reference(ids.tolist(),e,c,b)
   for field,values in want.items():
    w=np.asarray(values,dtype='i4').reshape(-1);got=np.fromfile(d/(field+'.bin'),dtype='i4');w.tofile(d/(field+'-cpu.bin'));check(d.name+'-'+field,got,w)
 # Gather changed routes and invalid/status mask with actual T513 input.
 for c in(64,128):
  t,r,b,slot=513,8,2,1;x=rng.integers(0,65536,(t,6144),dtype='u2')
  for changed in(0,1):
   mapping=((np.arange((slot+b+1)*c)*13+7*changed)%(t*r)).astype('i4');mapping[slot*c]=-1;mapping[(slot+1)*c+c-1]=t*r
   w=np.zeros((b,c,6144),dtype='u2')
   for local in range(b):
    for row in range(c):
     q=mapping[(slot+local)*c+row]
     if 0<=q<t*r:w[local,row]=x[q//r]
   name=f'gather-c{c}-changed{changed}';got=consumer(name,'gk_route_wide_tile_gather_bf16_v1',[x,mapping,status],w.shape,'u2',(r,c,slot));check(name,got,w)
  got=consumer(f'gather-c{c}-bad','gk_route_wide_tile_gather_bf16_v1',[x,mapping,np.ones(1,dtype='i4')],w.shape,'u2',(r,c,slot));check(f'gather-c{c}-bad',got,np.zeros_like(w))
 # Literal gate oracle is the original full ELF, not numpy exp approximation.
 for c in(64,128):
  b,slot=2,1;partial=rng.uniform(-16,16,(b,c,512)).astype('f4');valid=np.asarray([0,c-1,1,0],dtype='i4')
  for i in range(b):partial[i,valid[slot+i]:]=np.nan
  name=f'gate-c{c}';got=consumer(name,'gk_route_wide_tile_gate_bf16_v1',[partial,valid,status],(b,c,256),'u2',(slot,))
  old=consumer(name+'-old','gk_mxfp4_graph_gate_rows',[partial,np.arange(4,dtype='i4').reshape(1,-1),valid.reshape(1,-1)],got.shape,'u2',(slot,),old=True);check(name,got,old,padding_partial='NaN masked before load')
  for i in range(b):assert not got[i,valid[slot+i]:].any()
  got=consumer(name+'-bad','gk_route_wide_tile_gate_bf16_v1',[np.full_like(partial,np.nan),valid,np.ones(1,dtype='i4')],got.shape,'u2',(slot,));check(name+'-bad',got,np.zeros_like(old))
 # Padded rows beyond the old 32-row bound, all original route FMA order.
 libm=ctypes.CDLL('libm.so.6');assert libm.fegetround()==0;fma=libm.fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
 for c in(64,128):
  t,r,width=513,8,512;b=384+(t*r-384)//c;rows=b*c;partial=np.full((rows,width),np.nan,dtype='f4');routing=rng.uniform(-1,1,(t,r)).astype('f4');inv=rng.choice(rows,t*r,replace=False).astype('i4');values=rng.uniform(-8,8,(t*r,width)).astype('f4');partial[inv]=values
  want=np.zeros((t,width),dtype='f4')
  for token in range(t):
   for n in range(width):
    acc=0.
    for route in range(r):acc=fma(float(values[token*r+route,n]),float(routing[token,route]),acc)
    want[token,n]=acc
  name=f'combine-c{c}';got=consumer(name,'gk_route_wide_tile_combine_f32_v1',[partial,routing,inv,status],want.shape,'f4');check(name,got,want)
  bad=inv.copy();bad[0]=rows;got=consumer(name+'-bad-index','gk_route_wide_tile_combine_f32_v1',[partial,routing,bad,status],want.shape,'f4');assert np.isnan(got[0]).all();check(name+'-bad-index',got[1:],want[1:],nan_words=width)
 save('PASS_ACTUAL_ISA_WIDE_TILES',total_words=sum(r['words']for r in records))
except BaseException as e:save('FAIL',error=repr(e));raise
print(json.dumps(dict(status='PASS_ACTUAL_ISA_WIDE_TILES',checks=len(records),words=sum(r['words']for r in records))))
