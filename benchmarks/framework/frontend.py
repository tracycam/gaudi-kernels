"""Validate the public prepared-weight API, not only its internal node primitives."""
import json
import os
from pathlib import Path
import sys
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.torch_linear import load_extension,prepare_separable_fp8,linear_fp8
load_extension(root/'artifacts/builds/torch-c/gaudi_kernels_torch.so')
out=Path(os.environ['PROBE_OUT']);torch.set_num_threads(4);torch.manual_seed(29)


def sync():hc.mark_step();torch.hpu.synchronize()


codes=torch.arange(256,dtype=torch.uint8);codes=codes[(codes&127)!=127]
w=codes[None,:].view(torch.float8_e4m3fn)
p=prepare_separable_fp8(w,torch.ones(1))
assert p.weight.element_size()==1 and p.weight.numel()==w.numel()
assert int((p.weight.view(torch.uint8)&127).max())<120
# Subnormal-only channel remains byte-identical; mixed channels use explicit RNE.
tiny=torch.arange(16,dtype=torch.uint8)[None,:].view(torch.float8_e4m3fn)
assert torch.equal(prepare_separable_fp8(tiny,torch.ones(1)).weight.view(torch.uint8),tiny.view(torch.uint8))
rejected=0
for weight,scale in [(w,torch.ones(1,1)),(w,torch.tensor([-1.])),(torch.tensor([[127]],dtype=torch.uint8).view(torch.float8_e4m3fn),torch.ones(1)),(w,torch.tensor([3e38]))]:
    try:prepare_separable_fp8(weight,scale)
    except ValueError:rejected+=1
assert rejected==4
records=[]
for m,n,k in [(1,129,257),(16,128,256)]:
    original=codes[torch.randint(len(codes),(n,k))].view(torch.float8_e4m3fn)
    scales=torch.rand(n)*.02+.01;bias=torch.randn(n)*.1;x0=torch.randn(m,k).bfloat16()
    packed=prepare_separable_fp8(original,scales,bias);device=packed.to('hpu');x=x0.to('hpu');sync()
    original_ref=x0.double()@(original.double()*scales.double()[:,None]).T+bias.double()
    for policy in ['bf16','per_token_fp8']:
        graph=torch.hpu.HPUGraph();stream=torch.hpu.Stream()
        with torch.hpu.graph(graph,stream=stream):y=linear_fp8(x,device,activation=policy)
        sync();graph.replay();sync();actual=y.cpu().double()
        activ=x0.double()
        if policy=='per_token_fp8':
            sa=x0.float().abs().amax(-1,keepdim=True).clamp_min(1e-10)*(1/448)
            q=(x0.float()/sa).clamp(-448,448).to(torch.float8_e4m3fn).double()
            gpu=(q*sa.double())@(original.double()*scales.double()[:,None]).T+bias.double()
            activ=(q*.5).to(torch.float8_e4m3fn).double()*(sa.double()*2)
        else:gpu=original_ref
        contract=(activ@packed.weight.double().T*packed.scale.double()+bias.double()).bfloat16().double()
        error=float((actual-original_ref).norm()/original_ref.norm())
        gpu_error=float((gpu.bfloat16().double()-original_ref).norm()/original_ref.norm())
        contract_error=float((actual-contract).norm()/contract.norm())
        record={'M':m,'N':n,'K':k,'activation_policy':policy,'full_outputs':m*n,
                'relative_l2_to_original_fp64':error,'gpu_style_bf16_reference_error':gpu_error,
                'relative_l2_to_prepared_contract':contract_error,'device_weight_bytes':device.weight.numel(),
                'original_weight_bytes':original.numel(),'finite':bool(torch.isfinite(actual).all())}
        assert record['finite'] and contract_error<.001 and error<=max(.0035,gpu_error*1.1),record
        records.append(record);print(json.dumps(record),flush=True)
(out/'result.json').write_text(json.dumps({'status':'PASS','invalid_format_rejections':rejected,'records':records},indent=2)+'\n')
