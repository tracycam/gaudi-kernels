"""Bounded native recipe gate and timing without Python replay gaps."""
import argparse,json,os,subprocess,statistics
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--binary',required=True);a=p.parse_args();binary=str(Path(a.binary).resolve());out=Path(os.environ['PROBE_OUT']);torch.set_num_threads(4);torch.manual_seed(280931)
records=[]
for m,h in [(3,129),(1,6144),(64,6144),(513,6144)]:
 key=f'{m}-{h}';fixture=out/key;fixture.mkdir();x=torch.randn(m,h).bfloat16();r=torch.randn(m,h).bfloat16();w=(torch.randn(h)*.2+1).bfloat16()
 if h==129:x[0].zero_();r[0].zero_();r[1]=-x[1];r[1,::7]+=torch.tensor(.0078125,dtype=torch.bfloat16)
 for name,t in [('x',x),('residual',r),('gamma',w)]:t.view(torch.int16).numpy().tofile(fixture/(name+'.bin'))
 rr=(x+r).bfloat16();oracle=(rr.double()*torch.rsqrt(rr.double().square().mean(-1,keepdim=True)+float(torch.tensor(1e-6)))*w.double()).bfloat16();oracle.view(torch.int16).numpy().tofile(fixture/'oracle_fp64.bin')
 norm=None
 for mode in ['bf16','separate','per_row','block128']:
  directory=fixture/mode;directory.mkdir();env=dict(os.environ,ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(directory/'post_graph.json'),HABANA_LOGS=str(directory/'habana-logs'))
  command=[binary,mode,str(m),str(h),str(fixture),str(directory)]
  with (directory/'run.log').open('w') as log:run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=45)
  assert run.returncode==0,(key,mode,run.returncode)
  actual_r=torch.from_numpy(np.fromfile(directory/'residual_out.bin',np.uint16).view(np.int16).copy()).view(torch.bfloat16).view(m,h);assert torch.equal(actual_r,rr)
  record={'M':m,'H':h,'mode':mode,'command':command,'residual_bitwise':True}
  if mode=='bf16':
   norm=torch.from_numpy(np.fromfile(directory/'norm.bin',np.uint16).view(np.int16).copy()).view(torch.bfloat16).view(m,h)
   err=float((norm.double()-oracle.double()).norm()/oracle.double().norm().clamp_min(1e-30));assert err<.001;record['fp64_relative_l2']=err
  else:
   value=norm
   if mode=='block128':value=torch.nn.functional.pad(value,(0,(-h)%128)).view(m,-1,128).permute(1,0,2).contiguous()
   scale=value.float().abs().amax(-1,keepdim=True).clamp_min(1e-10)*torch.tensor(1/448,dtype=torch.float32)
   ref=((value.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*.5).to(torch.float8_e4m3fn)
   q=torch.from_numpy(np.fromfile(directory/'native_a8.bin',np.uint8).copy()).reshape(ref.shape)
   s=torch.from_numpy(np.fromfile(directory/'native_scale.bin',np.float32).copy()).reshape(scale.shape)
   ref.view(torch.uint8).numpy().tofile(directory/'expected_native.bin');(scale*2).numpy().tofile(directory/'expected_scale.bin')
   record['code_mismatch']=int((q!=ref.view(torch.uint8)).sum());record['scale_mismatch']=int((s.view(torch.int32)!=(scale*2).view(torch.int32)).sum());assert record['code_mismatch']==record['scale_mismatch']==0,record
  timing=[json.loads(line) for line in (directory/'run.log').read_text().splitlines() if line.startswith('{') and 'event_us' in line];assert len(timing)==5;record['timing']=timing;records.append(record);print(json.dumps(record),flush=True)
  (out/'result.json').write_text(json.dumps({'status':'RUNNING','records':records},indent=2)+'\n')
(out/'result.json').write_text(json.dumps({'status':'PASS','records':records,'scope':'full residual+norm(+quant) native recipe, QKV measured separately; no whole-model TPS claim'},indent=2)+'\n')
