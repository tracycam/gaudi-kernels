"""Full saved M513 device output gate and CPU reference cost; no device access."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from contract import reference,assess
p=argparse.ArgumentParser();p.add_argument('--saved',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'));sys.path.insert(0,str(root/'benchmarks/block_fp8_framework'))
from gaudi_kernels.block_fp8 import prepare_block_fp8
from oracle import adapted_weight
torch.set_num_threads(4);start=time.perf_counter();s=torch.load(a.saved,map_location='cpu',weights_only=False);layer=s['layers'][0];prep=prepare_block_fp8(layer['raw'],layer['scales'],layer['bias']);w=adapted_weight(prep,'fp32')
assert torch.equal(prep.weight.view(torch.uint8),layer['prepared_bytes'])and torch.equal(prep.scales,layer['prepared_scales'])
records=[]
for name,x,y in [('initial',s['x'],s['actual']),('changed',-s['x'],s['changed'])]:
 begin=time.perf_counter();r=reference(x,w,layer['bias']);got=assess(y,r);got.update(state=name,CPU_seconds=time.perf_counter()-begin);records.append(got);assert got['pass_all'],got
 wrong=(y.float()*1.01).bfloat16();assert not assess(wrong,r)['pass_all'],'1% scale error accepted'
result=dict(status='PASS_SAVED_FULL_M513_FP32_BOUND',records=records,shape=list(s['x'].shape),N=w.shape[0],total_CPU_wall_s=time.perf_counter()-start,saved_sha256=hashlib.sha256(a.saved.read_bytes()).hexdigest(),saved_path=str(a.saved.resolve()),device_accessed=False,scope='Prior native-lanes device output, same original and adapted representation, independent complete forward-error envelope; not a new padded device result.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
