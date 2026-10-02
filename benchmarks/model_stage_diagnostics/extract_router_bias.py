"""Read one small checkpoint correction-bias tensor on CPU; no accelerator calls."""
import argparse,hashlib,json
from pathlib import Path
from safetensors import safe_open
import torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--layer',type=int,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();assert 1<=a.layer<70
index=a.checkpoint/'model.safetensors.index.json'
key=f'model.layers.{a.layer}.mlp.gate.e_score_correction_bias'
filename=json.loads(index.read_text())['weight_map'][key]
path=(a.checkpoint/filename).resolve(strict=True)
assert path.is_relative_to(a.checkpoint.resolve(strict=True))
with safe_open(path,framework='pt',device='cpu')as source:value=source.get_tensor(key).clone()
assert value.shape==(384,) and value.device.type=='cpu' and value.isfinite().all()
record=dict(checkpoint=str(a.checkpoint),index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
    file=filename,key=key,dtype=str(value.dtype),shape=list(value.shape),
    tensor_bytes_sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    values=value.float().tolist(),device_accessed=False)
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('x')as file:file.write(json.dumps(record)+'\n')
