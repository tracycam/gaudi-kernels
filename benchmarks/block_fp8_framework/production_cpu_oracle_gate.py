"""Current production apply versus independent CPU A8 references on one QKV.

One device, M1. No model-quality or performance acceptance. The minimal loader
scaffold exercises the real production apply including CPU diagnostic readback.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import types

p=argparse.ArgumentParser();p.add_argument('--extension',required=True);p.add_argument('--fixture',type=Path,required=True);a=p.parse_args()
os.environ['GK_BLOCK_FP8_KEEP_CPU_ORACLE_WEIGHTS']='1'
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.block_fp8 import prepare_block_fp8
from gaudi_kernels.production_integration import install_block_fp8

class Upstream:
    def __init__(self,config):self.quant_config=config;self.block_quant=True
class Fp8LinearMethod(Upstream):pass
plugin=types.ModuleType('vllm_gaudi.ops.hpu_fp8');plugin.OrigFp8LinearMethod=Upstream
plugin.Fp8LinearMethod=Fp8LinearMethod;plugin.fp8=types.SimpleNamespace(Fp8LinearMethod=Fp8LinearMethod)
sys.modules[plugin.__name__]=plugin
state=install_block_fp8(plugin,extension_path=a.extension)
method=plugin.Fp8LinearMethod(types.SimpleNamespace(is_checkpoint_fp8_serialized=True,weight_block_size=(128,128)))
w=torch.frombuffer(bytearray((a.fixture/'weight.bin').read_bytes()),dtype=torch.uint8).reshape(3392,6144).view(torch.float8_e4m3fn)
s=torch.frombuffer(bytearray((a.fixture/'scales.bin').read_bytes()),dtype=torch.float32).reshape(27,48)
layer=types.SimpleNamespace(prefix='model.layers.0.self_attn.qkv_proj',_gk_block_fp8_selected=True,
    _gk_block_fp8=prepare_block_fp8(w,s).to('hpu'),_gk_oracle_weight=w,_gk_oracle_scales=s)
torch.set_num_threads(4);torch.manual_seed(291927)
out=Path(os.environ['PROBE_OUT']);records=[]
with torch.inference_mode():
    for rank_shape in ((1,6144),(1,1,6144)):
        x0=torch.randn(rank_shape).bfloat16();x=x0.to('hpu');hc.mark_step();torch.hpu.synchronize()
        values={}
        for policy in ('decode_a8_bf16_fp32','cpu_ocp_a8_bf16_fp32','cpu_native_a8_bf16_fp32'):
            state.set_policy(policy)
            y=method.apply(layer,x+torch.zeros_like(x));hc.mark_step();torch.hpu.synchronize()
            values[policy]=y.cpu()
        actual=values['decode_a8_bf16_fp32'].double()
        errors={name:float((actual-value.double()).norm()/value.double().norm()) for name,value in values.items()}
        assert all(value<.006 for value in errors.values()),errors
        torch.save({'x':x0,'outputs':values},out/('fixture-'+str(len(rank_shape))+'.pt'))
        records.append({'shape':rank_shape,'relative_l2':errors,'finite':bool(torch.isfinite(actual).all())})
report={'status':'PASS','scope':'single-device production apply/CPU reference plumbing; not full model or performance',
    'records':records,'state':state.snapshot()}
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
