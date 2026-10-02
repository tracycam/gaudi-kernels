"""Finite actual-ISA control/candidate gates; independent ordered CPU fmaf."""
import argparse,ctypes,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=('rolled','unroll8'),default='rolled');a=p.parse_args()
build=a.build.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);records=[]
libm=ctypes.CDLL('libm.so.6');assert libm.fegetround()==0
fma=libm.fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
directions=np.empty((2,256),dtype='u1')
for parity in range(2):
 for byte in range(256):
  lane=byte//4;directions[parity,byte]=(lane%16)//2+((lane//16)%2)*32+(128 if lane%2==parity else 0)
rng=np.random.default_rng(20260928)
def run(directory,variant,partial,routing):
 d=directory/variant;d.mkdir();name={'control':'gk_moe_combine_scalar_control_experiment','candidate':'gk_moe_combine_vector_routes'+('_unroll8'if a.variant=='unroll8'else'')+'_experiment'}[variant]
 spec=[name]
 for i,x in enumerate([partial,routing,directions]):
  path=d/f'input{i}.bin';x=np.ascontiguousarray(x);x.tofile(path);spec.extend(['u8'if x.dtype==np.uint8 else'f32','2',*map(str,x.shape[::-1]),str(path)])
 t=routing.shape[0];h=partial.size//(t*8);path=d/'output.bin';spec.extend(['f32','2',str(h),str(t),str(path)])
 (d/'spec.txt').write_text(' '.join(spec)+'\n');command=[str(build/'simulator'),str(build/'libgaudi_combine_preload_tpc.so'),str(d/'spec.txt')]
 (d/'command.json').write_text(json.dumps(command)+'\n')
 with(d/'stdout.log').open('w')as f:r=subprocess.run(command,env={**os.environ,'TPC_RUNNER':'0'},cwd=d,stdout=f,stderr=subprocess.STDOUT,timeout=45)
 if r.returncode:raise RuntimeError(f'{variant} simulator exit {r.returncode}')
 stats=[json.loads(line)for line in(d/'stdout.log').read_text().splitlines()if line.startswith('{')][-1]
 return np.fromfile(path,dtype='f4').reshape(t,h),stats
try:
 for t,h,mode in [(1,128,'ordinary'),(2,256,'ordinary'),(8,6144,'ordinary'),(1,6144,'edge_bits'),(2,128,'cancellation')]:
  d=out/f't{t}-h{h}-{mode}';d.mkdir();natural=rng.normal(size=(t,8,h)).astype('f4');routing=rng.uniform(-1,1,(t,8)).astype('f4')
  if mode=='edge_bits':
   natural.flat[:]=np.resize(np.array([0.,-0.,2**-149,-2**-149,2**-126,-2**-126,2**90,-2**90,1.,-1.],dtype='f4'),natural.size)
   routing[:]=np.array([0.,-0.,1.,-1.,2**-126,2**-140,2**-10,2**10],dtype='f4')
  if mode=='cancellation':
   natural[:]=0;routing[:]=1;natural[:,0,:]=2**24;natural[:,1,:]=1;natural[:,2,:]=-2**24;natural[:,3,:]=1
  chunks=natural.reshape(t,8,h//128,128)
  partial=np.concatenate((chunks[...,::2],chunks[...,1::2]),axis=-1).reshape(1,-1)
  natural.tofile(d/'natural.bin');routing.tofile(d/'routing.bin')
  old,old_stats=run(d,'control',partial,routing);new,new_stats=run(d,'candidate',partial,routing)
  bad=int(np.count_nonzero(old.view('u4')!=new.view('u4')))
  row=dict(case=d.name,words=old.size,old_new_bit_mismatches=bad,control=old_stats,candidate=new_stats,
           scope='ISA equivalence only; no hardware or throughput claim')
  records.append(row);(out/'partial.json').write_text(json.dumps(records,indent=2)+'\n')
  if bad:raise AssertionError((d.name,bad))
  if mode!='edge_bits':
   oracle=np.empty((t,h),dtype='f4')
   for token in range(t):
    for column in range(h):
     acc=0.
     for slot in range(8):acc=fma(float(natural[token,slot,column]),float(routing[token,slot]),acc)
     oracle[token,column]=np.float32(acc)+np.float32(0) # original deinterleave adds its zero-filled partner
   oracle.tofile(d/'cpu-fmaf.bin');row['cpu_fmaf_bit_mismatches']=int(np.count_nonzero(old.view('u4')!=oracle.view('u4')))
   if row['cpu_fmaf_bit_mismatches']:raise AssertionError(('independent CPU FMA',row))
  else:row['CPU_scope']='signed-zero/subnormal/native-FTZ transport compared only to unchanged deployed ELF'
 report=dict(status='PASS_ACTUAL_ISA_ONLY',records=records,variant=a.variant,device_tested=False,production_changed=False,
             words=sum(r['words'] for r in records),source_build=json.loads((build/'build.json').read_text())['source_commit'])
except Exception as exc:report=dict(status='FAIL',records=records,error=repr(exc),device_tested=False);raise
finally:
 report['sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()for p in out.rglob('*')if p.is_file()}
 report['libraries']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [build/'libgaudi_combine_preload_tpc.so',Path('/usr/lib/habanatools/libtpc_tests_core_ext.so')]}
 (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ('records','sha256','libraries')}))
