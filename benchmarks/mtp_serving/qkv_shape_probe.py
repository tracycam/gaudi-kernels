"""Real layer5 QKV, same first input, M1 vs M4/8/12, no full-model reload."""
import hashlib,json,os,sys
from pathlib import Path
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from safetensors import safe_open
root=Path(__file__).resolve().parents[2];out=Path(os.environ['PROBE_OUT']);sys.path.insert(0,str(root/'python'))
from gaudi_kernels.block_fp8 import prepare_block_fp8,linear_block_fp8_quantized
for name in ('norm','grid','block','reduce'):torch.ops.load_library(str(root/'libraries'/f'{name}_torch.so'))
data=torch.load(root/'layer5-input.pt',map_location='cpu',weights_only=True)
model=Path('/data/models/MiMo-V2.6-Pro-RL');cfg=json.loads((model/'config.json').read_text());mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
def tensor(key,rows=None):
    with safe_open(model/mapping[key],framework='pt',device='cpu')as f:
        return (f.get_tensor(key)if rows is None else f.get_slice(key)[:rows]).contiguous()
gamma=tensor('model.layers.5.input_layernorm.weight')
prefix='model.layers.5.self_attn.qkv_proj.'
# TP8 rank0 owns one checkpoint Q/K/V group; its scale grid resets at row0.
weight=tensor(prefix+'weight',3392);scales=tensor(prefix+'weight_scale_inv',27)
def sync():hc.mark_step();torch.hpu.synchronize()
def comparison(a,b):
    a,b=a.cpu().contiguous(),b.cpu().contiguous()
    return dict(bits_equal=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),max_abs=float((a.float()-b.float()).abs().max()),
                mismatch_elements=int((a.float()!=b.float()).sum()))
records=[];outputs={}
with torch.inference_mode():
    prepared=prepare_block_fp8(weight,scales).to('hpu');g=gamma.to('hpu');sync()
    for rows in (1,4,8,12):
        x=data['x'].repeat(rows,1);r=data['residual'].repeat(rows,1)
        if rows>1:x[1:]*=3;r[1:]*=.5
        xd,rd=x.to('hpu'),r.to('hpu');sync()
        if rows==1:_,q,sa=torch.ops.gaudi_norm_grid24.block128(xd,rd,g,cfg['layernorm_epsilon'])
        else:
            _,norm=torch.ops.gaudi_kernels._residual_rmsnorm_bf16(xd,rd,g,cfg['layernorm_epsilon']);q,sa=torch.ops.gaudi_block_fp8.quant(norm)
        y=linear_block_fp8_quantized(q,sa,prepared,reduce_op=torch.ops.gk_reduce_isa.handschedule);sync();plain=y.cpu()
        witness={};yy=linear_block_fp8_quantized(q,sa,prepared,reduce_op=torch.ops.gk_reduce_isa.handschedule,audit_tensors=witness);sync()
        assert torch.equal(plain,yy.cpu()),'keeping intermediates changed final recipe'
        row={k:witness[k].cpu()[:,:1]for k in ('q_native','activation_scales','partial')};row['output']=plain[:1]
        outputs[str(rows)]=row
        if rows==1:baseline=row
        records.append(dict(rows=rows,comparisons={k:comparison(baseline[k],row[k])for k in row}))
    torch.save(outputs,out/'tensors.pt',pickle_protocol=2)
(out/'result.json').write_text(json.dumps(dict(status='COMPLETED_QKV_SHAPE_DIAGNOSTIC',records=records,
    fixture_sha256=hashlib.sha256((root/'layer5-input.pt').read_bytes()).hexdigest(),
    scope='Real TP8 rank0 layer5 norm/quant/FP8 MME/Neumaier; excludes attention, other ranks and model TPS'),indent=2)+'\n')
