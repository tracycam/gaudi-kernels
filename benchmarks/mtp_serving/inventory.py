"""Read safetensors headers, not weight payloads, to identify the real drafters."""
import argparse,hashlib,json,re,struct
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
records=[]
for name in ('model_mtp.safetensors','dflash/dflash_draft_model.safetensors'):
    path=a.model/name
    with path.open('rb')as f:
        length=struct.unpack('<Q',f.read(8))[0];assert length<64*1024*1024
        raw=f.read(length);header=json.loads(raw)
    tensors={k:v for k,v in header.items()if k!='__metadata__'};by_dtype=Counter();layers=set()
    for k,v in tensors.items():
        by_dtype[v['dtype']]+=v['data_offsets'][1]-v['data_offsets'][0]
        match=re.search(r'(?:^|\.)layers\.(\d+)\.',k)
        if match:layers.add(int(match[1]))
    records.append(dict(file=name,file_bytes=path.stat().st_size,header_sha256=hashlib.sha256(raw).hexdigest(),layers=sorted(layers),
        payload_bytes_by_dtype=dict(by_dtype),tensors=tensors))
result=dict(scope='Checkpoint identity/storage audit. No acceptance rate or performance claim.',records=records,
    dflash_config=json.loads((a.model/'dflash/config.json').read_text()))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
for r in records:print({k:v for k,v in r.items()if k!='tensors'})
