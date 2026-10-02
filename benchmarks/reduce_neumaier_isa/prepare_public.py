"""CPU-only inputs/oracles for the public-graph gate; no runtime acquisition."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--checkpoint-witness',type=Path,required=True);p.add_argument('--activation-witness',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'python'),str(root/'benchmarks/block_fp8_framework')]
from gaudi_kernels.block_fp8 import prepare_block_fp8
from oracle import reference,quantize_block
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(4);torch.manual_seed(280927)
shutil.copytree(a.fixtures,a.out/'reduction')
records=[]
for name,m,n,k in [('qkv-measured-m1',1,3392,6144),('tail-m3',3,257,513),('single-group-m16',16,128,128)]:
 if name=='qkv-measured-m1':
  raw=torch.from_numpy(np.fromfile(a.checkpoint_witness/'weight-e4m3.bin',np.uint8).copy()).view(torch.float8_e4m3fn).reshape(n,k)
  scales=torch.from_numpy(np.fromfile(a.checkpoint_witness/'scales-f32.bin',np.float32).copy()).reshape(27,48)
  x=torch.from_numpy(np.fromfile(a.activation_witness,np.uint16).copy().view(np.int16)).view(torch.bfloat16).reshape(m,k)
  bias=torch.zeros(n)
 else:
  raw=(torch.randn(n,k)*64).clamp(-448,448).to(torch.float8_e4m3fn);scales=torch.rand((n+127)//128,(k+127)//128)*.0002+.0001;x=torch.randn(m,k).bfloat16();bias=torch.randn(n)*.01
 prep=prepare_block_fp8(raw,scales,bias);samples=[]
 for xx in [x,-x]:
  q,sa,_,_=quantize_block(xx);refs=reference(xx,raw,scales,bias,prep,'per_block_fp8')
  samples.append(dict(x=xx,quant_bits=q.view(torch.uint8),activation_scales=sa,refs=refs))
 fixture=dict(name=name,shape=[m,n,k],weight_bits=prep.weight.view(torch.uint8),weight_scales=prep.scales,bias=prep.bias,samples=samples,checkpoint_weight_sha256=prep.checkpoint_weight_sha256,checkpoint_scales_sha256=prep.checkpoint_scales_sha256)
 path=a.out/(name+'.pt');torch.save(fixture,path);records.append(dict(name=name,shape=[m,n,k],file=path.name,sha256=sha(path)))
(a.out/'chain-plan.json').write_text(json.dumps(dict(device_tested=False,records=records),indent=2)+'\n')
files={str(path.relative_to(a.out)):sha(path) for path in sorted(a.out.rglob('*')) if path.is_file()}
(a.out/'fixture-identity.json').write_text(json.dumps(dict(device_tested=False,files_sha256=files),indent=2)+'\n')
print(json.dumps(dict(chain_cases=len(records),reduction_cases=len(json.loads((a.out/'reduction/summary.json').read_text())['records']),files=len(files),device_accessed=False)))
