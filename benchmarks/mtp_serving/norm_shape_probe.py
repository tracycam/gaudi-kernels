"""Isolate the real first-divergence layer's norm/quant without reloading 70 layers."""
import hashlib,json,os
from pathlib import Path
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from safetensors import safe_open

root=Path(__file__).resolve().parents[2];out=Path(os.environ['PROBE_OUT'])
config=json.loads((root/'norm-probe.json').read_text())
for name in ('norm','grid','block'):torch.ops.load_library(str(root/'libraries'/f'{name}_torch.so'))
data=torch.load(root/'layer5-input.pt',map_location='cpu',weights_only=True)
model=Path('/data/models/MiMo-V2.6-Pro-RL');cfg=json.loads((model/'config.json').read_text())
key='model.layers.5.input_layernorm.weight';mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
with safe_open(model/mapping[key],framework='pt',device='cpu')as f:gamma=f.get_tensor(key)
def sync():hc.mark_step();torch.hpu.synchronize()
def compare(a,b):
    return dict(bits_equal=torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),
                max_abs=float((a.float()-b.float()).abs().max()))
records=[];values={}
with torch.inference_mode():
    g=gamma.to('hpu');x=data['x'].to('hpu');r=data['residual'].to('hpu');sync()
    rr,qq,ss=torch.ops.gaudi_norm_grid24.block128(x,r,g,cfg['layernorm_epsilon']);sync()
    baseline=[v.cpu()for v in (rr,qq,ss)];values['grid']=baseline
    for rows in (1,4,8,12):
        # Different other rows must never change the first row's arithmetic.
        xx=data['x'].repeat(rows,1);rr=data['residual'].repeat(rows,1)
        if rows>1:xx[1:]*=3;rr[1:]*=.5
        xd,rd=xx.to('hpu'),rr.to('hpu');sync()
        residual,norm=torch.ops.gaudi_kernels._residual_rmsnorm_bf16(xd,rd,g,cfg['layernorm_epsilon'])
        q,scale=torch.ops.gaudi_block_fp8.quant(norm);sync()
        actual=[residual.cpu()[:1],q.cpu()[:,:1],scale.cpu()[:,:1]]
        values[str(rows)]=dict(residual=actual[0],q=actual[1],scale=actual[2],norm=norm.cpu()[:1])
        records.append(dict(rows=rows,comparisons={k:compare(a,b)for k,a,b in zip(('residual','q','scale'),baseline,actual)}))
    torch.save(dict(inputs=data,gamma=gamma,values=values),out/'tensors.pt',pickle_protocol=2)
(out/'result.json').write_text(json.dumps(dict(status='COMPLETED_NORM_SHAPE_DIAGNOSTIC',records=records,
    epsilon=cfg['layernorm_epsilon'],fixture_sha256=hashlib.sha256((root/'layer5-input.pt').read_bytes()).hexdigest(),
    scope='Same real layer5 input: M1 grid24 vs separate norm+quant M1/4/8/12; does not test QKV MME or attention'),indent=2)+'\n')
