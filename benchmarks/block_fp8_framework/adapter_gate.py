"""Offline loader lifecycle regression with a minimal upstream-method stub.

This does not claim a vLLM service/model run; it proves no Gaudi loader double
adaptation is inserted, parameters are replaced, and repeated prep is refused.
"""
import json
from pathlib import Path
import sys
import types
import torch
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.block_fp8 import prepare_block_fp8,linear_block_fp8
from gaudi_kernels.vllm_block_fp8 import make_vllm_block_fp8_method
calls=[]
class Upstream:
    def __init__(self,config):self.quant_config=config;self.block_quant=True
    def create_weights(self,*a,**kw):calls.append('upstream')
stub=types.ModuleType('vllm_gaudi.ops.hpu_fp8');stub.OrigFp8LinearMethod=Upstream
sys.modules['vllm_gaudi.ops.hpu_fp8']=stub
config=types.SimpleNamespace(is_checkpoint_fp8_serialized=True,weight_block_size=[128,128])
method=make_vllm_block_fp8_method(activation='bf16')(config);method.create_weights()
layer=torch.nn.Module();raw=torch.ones((129,129)).to(torch.float8_e4m3fn);scales=torch.ones(2,2)
layer.weight=torch.nn.Parameter(raw.clone(),requires_grad=False)
layer.weight_scale_inv=torch.nn.Parameter(scales.clone(),requires_grad=False)
old_owner=layer.weight
method.process_weights_after_loading(layer)
assert calls==['upstream'] and layer.weight is not old_owner
assert layer._gk_block_fp8.weight is layer.weight and layer._gk_block_fp8.scales is layer.weight_scale_inv
assert torch.equal(layer.weight.float(),torch.full_like(layer.weight.float(),.5).where(layer.weight.float()!=0,0))
assert torch.equal(layer.weight_scale_inv,scales*2) and torch.equal(old_owner,raw)
rejected=0
try:method.process_weights_after_loading(layer)
except ValueError:rejected+=1
# A finite source scale need not yield finite BF16 decoded weights. Do not
# admit that decode plan even if an A8 dot could have finite cancellation.
wide=prepare_block_fp8(torch.full((1,1),448.).to(torch.float8_e4m3fn),torch.full((1,1),1e37))
assert not wide.bf16_fp32_scale_safe and not wide.bf16_bf16_scale_safe
try:linear_block_fp8(torch.ones(1,1,dtype=torch.bfloat16),wide,activation='bf16')
except ValueError:rejected+=1
assert rejected==2
print(json.dumps({'status':'PASS_CPU_STUB_LIFECYCLE','upstream_create_calls':len(calls),'rejected_double_prep_and_bf16_overflow':rejected,'actual_vllm_service_validated':False}))
