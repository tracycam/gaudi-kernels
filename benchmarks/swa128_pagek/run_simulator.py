import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.manual_seed(928128);q=(torch.randn(2,192)*.5).bfloat16();k=(torch.randn(3,128,192)*.5).bfloat16()
q[1,:]=1;q[1,1::2]=-1
k[:,:,1::2]=k[:,:,::2];k[:,:,-1]+=torch.tensor(2**-8,dtype=torch.bfloat16)
kt=k.transpose(1,2).contiguous();records=[]
for name,page,lo,count in [('full',2,0,128),('partial31',0,17,31),('last1',1,127,1),('invalid',-1,0,0)]:
 out=a.output/name;out.mkdir();q.view(torch.uint16).numpy().tofile(out/'q.bin');k.view(torch.uint16).numpy().tofile(out/'k_row.bin');kt.view(torch.uint16).numpy().tofile(out/'kt.bin');np.array([page,lo,count],np.int32).tofile(out/'metadata.bin')
 ref=np.zeros((2,128),np.float64);full=np.zeros_like(ref)
 if page>=0:
  products=q.double().numpy()[:,None,:]*k[page,lo:lo+count].double().numpy()[None,:,:];ref[:,:count]=products.sum(-1);full[:,:count]=np.abs(products).sum(-1)
 ref.tofile(out/'reference-f64.bin');full.tofile(out/'fullmac-f64.bin')
 command=[str((a.build/'simulator').resolve()),str((a.build/'qk.o').resolve()),str(out.resolve()),str((out/'actual-f32.bin').resolve())]
 completed=subprocess.run(command,env={**os.environ,'TPC_RUNNER':'0'},capture_output=True,timeout=45);(out/'run.log').write_bytes(completed.stdout+completed.stderr)
 if completed.returncode:raise RuntimeError((name,completed.returncode))
 actual=np.fromfile(out/'actual-f32.bin',np.float32).reshape(2,128)
 variants={'linear':actual,'even_odd_interleaved':np.stack([actual[:,i]for i in sum(([i,i+64]for i in range(64)),[])],axis=1)}
 errors={n:float(np.max(np.abs(v-ref)))for n,v in variants.items()}
 linear=variants['even_odd_interleaved'];normal_error=np.max(np.abs(linear-ref)/np.maximum(full,1e-30))
 assert np.isfinite(actual).all() and normal_error<4e-6,(name,normal_error)
 linear.tofile(out/'actual-linearized-f32.bin')
 record={'case':name,'errors_by_lane_order':errors,'fullmac_normalized_max_error':float(normal_error),'finite':bool(np.isfinite(actual).all()),'raw_cycle_log':completed.stdout.decode(),'device_verified':False};records.append(record)
 print(json.dumps(record),flush=True)
(a.output/'result.json').write_text(json.dumps({'status':'PASS_NATIVE_EVEN_ODD_SCORE_LAYOUT','records':records,'elf_sha256':hashlib.sha256((a.build/'qk.o').read_bytes()).hexdigest(),'scope':'score-only original BF16 Q/K and FP32 MAC; no query scale/softmax/AV/writer/graph/device qualification'},indent=2)+'\n')
