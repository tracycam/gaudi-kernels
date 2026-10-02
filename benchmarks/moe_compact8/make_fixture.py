"""Explicit large immutable fixture generation; run only after model timing ends.

No device access. Writes 962,592,768 original-width expert bytes. No BF16/FP32
weight cache is created. The seed is synthetic; this is production geometry and
lossless production packing, not a checkpoint/model quality fixture.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from ledger import E,SHAPES,routes,accounting

p=argparse.ArgumentParser();p.add_argument('--runtime-fixture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-large-fixture',action='store_true')
a=p.parse_args()
if not a.allow_large_fixture:p.error('Explicit --allow-large-fixture required; do not run during parent model timing')
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
packing=a.runtime_fixture/'executor/kernels/packing.py'
spec=importlib.util.spec_from_file_location('fixture_packing',packing);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert hashlib.sha256(packing.read_bytes()).hexdigest()=='cf69f2f96054d1d6cc35ea3dc7390bb55c3394403a408e16e177287c438d9457'
rng=np.random.default_rng(928384);torch.set_num_threads(2);torch.manual_seed(928384)
record={'layout':'historical_N512','experts':E,'seed':928384,'scale_codes':[115,123],
        'source':'synthetic values; exact TP8 production shapes and lossless production packing',
        'weights':{},'cases':[],'status':'WRITING'}
for n,k,wn,sn in [(512,6144,'gp','gs'),(6144,256,'dp','ds')]:
    weight=np.memmap(out/(wn+'.bin'),mode='w+',dtype=np.uint8,shape=SHAPES[wn])
    scale=np.memmap(out/(sn+'.bin'),mode='w+',dtype=np.uint8,shape=SHAPES[sn])
    wr,sr=SHAPES[wn][0]//E,SHAPES[sn][0]//E
    for expert in range(E):
        original=rng.integers(0,256,(n,k//2),dtype=np.uint8);scales=rng.integers(115,124,(n,k//32),dtype=np.uint8)
        w,s=module.pack(original,scales)
        # Every expert's source bytes are recovered, including expert383.
        iw,isc=module.unpack(w,s);assert np.array_equal(iw,original) and np.array_equal(isc,scales)
        weight[expert*wr:(expert+1)*wr]=w.reshape(wr,256)
        scale[expert*sr:(expert+1)*sr]=s.reshape(sr,512)
    weight.flush();scale.flush();del weight,scale
for name,shape in SHAPES.items():
    h=hashlib.sha256()
    with(out/(name+'.bin')).open('rb')as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    record['weights'][name]={'shape':shape,'dtype':'uint8','bytes':int(np.prod(shape)),'sha256':h.hexdigest()}
for tokens in (2,8):
    x=(torch.randn(tokens,6144)*.1).bfloat16()
    for state in ('sparse','hot','rotated','zero'):
        ids=routes(tokens,state);weights=torch.rand(tokens,8,dtype=torch.float32);weights/=weights.sum(-1,keepdim=True)
        if state=='zero':weights.zero_()
        name=f't{tokens}-{state}';path=out/(name+'.pt')
        torch.save({'x':x if state in ('sparse','zero') else (-x if state=='rotated' else x.roll(17,-1)),
                    'ids':torch.tensor(ids,dtype=torch.int32),'routing':weights},path)
        ledger=accounting(ids);(out/(name+'-ledger.json')).write_text(json.dumps(ledger,indent=2)+'\n')
        record['cases'].append({'name':name,'tokens':tokens,'state':state,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
record['status']='IMMUTABLE_FIXTURE_READY'
(out/'fixture.json').write_text(json.dumps(record,indent=2)+'\n')
