"""CPU evidence for bytes, independent scales, and split-view layout only."""
import json
from pathlib import Path
import sys
import torch

root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.fp8_projections import prepare_fp8_projections,split_fp8_projections
from gaudi_kernels.torch_linear import prepare_separable_fp8
torch.manual_seed(270927);torch.set_num_threads(2)
codes=torch.arange(256,dtype=torch.uint8);codes=codes[(codes&127)!=127]
records=[]
for widths,k,m in [((17,31,65),33,1),((129,64,257),259,7),((1024,512,256),2048,64)]:
    weights=[codes[torch.randint(codes.numel(),(n,k))].view(torch.float8_e4m3fn) for n in widths]
    # Keep one subnormal-only channel and one extended-code channel distinct.
    weights[0].view(torch.uint8)[0]=torch.arange(k,dtype=torch.int64).remainder(16).to(torch.uint8)
    originals=[w.view(torch.uint8).clone() for w in weights]
    scales=[torch.tensor(.125),torch.rand(widths[1]),torch.rand(widths[2])]
    biases=[None,torch.randn(widths[1]),torch.randn(widths[2])]
    prepared=prepare_fp8_projections(weights,scales,biases)
    parts=[prepare_separable_fp8(w,s,b) for w,s,b in zip(weights,scales,biases)]
    start=0
    for original,w,p,n in zip(originals,weights,parts,widths):
        assert torch.equal(w.view(torch.uint8),original)
        assert torch.equal(prepared.linear.weight.view(torch.uint8)[start:start+n],p.weight.view(torch.uint8))
        assert torch.equal(prepared.linear.scale[start:start+n],p.scale)
        assert torch.equal(prepared.linear.bias[start:start+n],p.bias)
        start+=n
    assert torch.equal(prepared.linear.weight.view(torch.uint8)[0],originals[0][0])
    x=torch.randn(m,k).bfloat16().double()
    combined=x@prepared.linear.weight.double().T*prepared.linear.scale.double()+prepared.linear.bias.double()
    expected=torch.cat([x@p.weight.double().T*p.scale.double()+p.bias.double() for p in parts],1)
    assert torch.allclose(combined,expected,rtol=1e-12,atol=1e-10)
    views=split_fp8_projections(combined,prepared)
    assert all(v.untyped_storage().data_ptr()==combined.untyped_storage().data_ptr() for v in views)
    dense=split_fp8_projections(combined,prepared,contiguous=True)
    assert all(v.is_contiguous() for v in dense)
    records.append({'M':m,'K':k,'widths':list(widths),'weight_bytes':prepared.linear.weight.numel(),
                    'original_weight_bytes':sum(w.numel() for w in weights),
                    'separate_preparation_bitwise_equal':True,'shared_output_storage':True,
                    'views_contiguous':[v.is_contiguous() for v in views]})
try:prepare_fp8_projections(weights,[torch.ones(1,1)]*3)
except ValueError:pass
else:raise AssertionError('block-shaped scale accepted')
try:prepare_fp8_projections([weights[0],weights[1][:,:1].contiguous()],scales[:2])
except ValueError:pass
else:raise AssertionError('different activation K accepted')
print(json.dumps({'status':'PASS','scope':'CPU packing and mathematical identity; no HPU performance or placement claim',
                  'records':records},indent=2))
