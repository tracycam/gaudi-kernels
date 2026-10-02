"""CPU structural gate for selective production installation; no HPU/vLLM run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types
import torch

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
subprocess.run([sys.executable,'-c',
    f"import sys;sys.path.insert(0,{str(root/'python')!r});import gaudi_kernels.production_integration as p;assert 'torch' not in sys.modules;assert 'vllm_gaudi' not in sys.modules;assert p.snapshot()=={{'installed':False}}"],check=True)
import gaudi_kernels.production_integration as pi
import gaudi_kernels.block_fp8 as block
torch.set_num_threads(2)
calls=[];operator_calls=[];external_calls=[]


class Upstream:
    def __init__(self,config):self.quant_config=config;self.block_quant=config.weight_block_size is not None
    def create_weights(self,layer,*,weight_loader):
        calls.append(('upstream_create',layer.prefix,weight_loader))
        layer.weight=torch.nn.Parameter(torch.ones(layer.shape).to(torch.float8_e4m3fn),requires_grad=False)
        n,k=layer.shape
        layer.weight_scale_inv=torch.nn.Parameter(torch.ones((n+127)//128,(k+127)//128),requires_grad=False)
    def process_weights_after_loading(self,layer):raise AssertionError('the upstream postprocess is not this adapter')


class Fp8LinearMethod(Upstream):
    def create_weights(self,layer,**kwargs):
        calls.append(('gaudi_create',layer.prefix));return super().create_weights(layer,**kwargs)
    def process_weights_after_loading(self,layer):calls.append(('gaudi_postprocess',layer.prefix))
    def apply(self,layer,x,bias=None):calls.append(('gaudi_apply',layer.prefix));return x
    def dequant_fp8_weight(self,layer):return layer.weight


def module():
    m=types.ModuleType('vllm_gaudi.ops.hpu_fp8');m.OrigFp8LinearMethod=Upstream
    m.Fp8LinearMethod=Fp8LinearMethod;m.fp8=types.SimpleNamespace(Fp8LinearMethod=Fp8LinearMethod)
    sys.modules[m.__name__]=m
    return m


def layer(prefix,shape,tp=8):
    value=torch.nn.Module();value.prefix=prefix;value.shape=shape;value.tp_size=tp
    return value


def config(blocksize=(128,128)):
    return types.SimpleNamespace(is_checkpoint_fp8_serialized=True,weight_block_size=blocksize)


def fake_linear(x,prepared,**kw):
    operator_calls.append((x,prepared,kw));return torch.zeros(x.shape[0],prepared.n,dtype=torch.bfloat16)


def fake_quantized(q,scales,prepared,**kw):
    external_calls.append((q,scales,prepared,kw));return torch.zeros(q.shape[1],prepared.n,dtype=torch.bfloat16)


block.linear_block_fp8=fake_linear;block.linear_block_fp8_quantized=fake_quantized
# Schemas only: the gate cannot execute device kernels accidentally.
schemas=torch.library.Library('gaudi_block_fp8','DEF')
for name in ('quant','batch_mm','reduce','decode','decode_fast','mm','finish'):
    schemas.define(f'{name}(Tensor x) -> Tensor')
schemas.define('reshape(Tensor x, int[] shape) -> Tensor')
reshape_cpu=torch.library.Library('gaudi_block_fp8','IMPL','CPU')
reshape_cpu.impl('reshape',lambda x,shape:x.reshape(shape))
disabled=module();assert pi.install_block_fp8(disabled,enabled=False) is None
assert disabled.Fp8LinearMethod is Fp8LinearMethod
plugin=module();handle=pi.install_block_fp8(plugin)
assert plugin.Fp8LinearMethod is plugin.fp8.Fp8LinearMethod
assert plugin.Fp8LinearMethod.__name__=='Fp8LinearMethod'
assert pi.install_block_fp8(plugin) is handle

loader=object();qkv=layer('model.layers.0.self_attn.qkv_proj',(3392,6144))
method=plugin.Fp8LinearMethod(config());method.create_weights(qkv,weight_loader=loader)
assert calls[-1]==('upstream_create',qkv.prefix,loader)
assert not any(c[0]=='gaudi_create' for c in calls)
method.process_weights_after_loading(qkv)
prepared=qkv._gk_block_fp8;owner=prepared.weight
assert tuple(owner.shape)==(48,3392,128) and tuple(prepared.scales.shape)==(27,48)
assert torch.equal(owner.float(),torch.full(owner.shape,.5))
assert torch.equal(prepared.scales,torch.full((27,48),2.))
assert handle.snapshot()['selected_layer_count']==1

# Unselected block MLP, ordinary FP8, unsupported block shape and lookalike
# suffix all delegate every lifecycle stage to the existing Gaudi class.
fallbacks=[]
for prefix,cfg in [('model.layers.0.mlp.gate_proj',config()),
                   ('model.layers.1.self_attn.qkv_proj',config(None)),
                   ('model.layers.2.self_attn.qkv_proj',config((64,128))),
                   ('model.layers.3.self_attn.not_qkv_proj',config())]:
    w=layer(prefix,(8,8));m=plugin.Fp8LinearMethod(cfg);m.create_weights(w,weight_loader=loader)
    m.process_weights_after_loading(w);x=torch.ones(1,8);assert m.apply(w,x) is x
    assert not hasattr(w,'_gk_block_fp8');fallbacks.append(prefix)

x2=torch.zeros(1,6144,dtype=torch.bfloat16)
y2=method.apply(qkv,x2);assert operator_calls[-1][0] is x2 and tuple(y2.shape)==(1,3392)
assert operator_calls[-1][2]['activation']=='bf16' and operator_calls[-1][2]['scale_math']=='fp32'
x3=torch.zeros(1,1,6144,dtype=torch.bfloat16)
y3=method.apply(qkv,x3);assert tuple(y3.shape)==(1,1,3392)
initial=handle.snapshot()
pi.set_policy('decode_a8_bf16_fp32')
method.apply(qkv,x2);assert operator_calls[-1][2]['activation']=='per_block_fp8'
method.apply(qkv,torch.zeros(2,6144,dtype=torch.bfloat16));assert operator_calls[-1][2]['activation']=='bf16'
method.apply(qkv,x3);assert operator_calls[-1][2]['activation']=='per_block_fp8'
q=torch.empty(48,1,128,dtype=torch.float8_e4m3fn);s=torch.ones(48,1,1)
assert tuple(method.apply_quantized(qkv,q,s).shape)==(1,3392) and len(external_calls)==1
assert qkv._gk_block_fp8.weight is owner
mixed=handle.snapshot();pi.set_policy('bf16_fp32');assert handle.generation==2

rejected=[]
def reject(name,fn):
    try:fn()
    except (ValueError,RuntimeError):rejected.append(name)
    else:raise AssertionError(name+' was accepted')
reject('quantized_input_under_A16',lambda:method.apply_quantized(qkv,q,s))
reject('unsupported_policy',lambda:pi.set_policy('auto_threshold_16'))
reject('repeated_preparation',lambda:method.process_weights_after_loading(qkv))
reject('expanded_weight_cache',lambda:method.dequant_fp8_weight(qkv))
reject('live_disable',lambda:pi.install_block_fp8(plugin,enabled=False))
reject('implicit_policy_change',lambda:pi.install_block_fp8(plugin,policy='decode_a8_bf16_fp32'))
wrong=layer('model.layers.4.self_attn.qkv_proj',(128,128));other=plugin.Fp8LinearMethod(config())
other.create_weights(wrong,weight_loader=loader)
reject('wrong_QKV_local_shape',lambda:other.process_weights_after_loading(wrong))

# Explicit all-block scope still preserves ordinary FP8; no prefix heuristics
# can silently enable it in the default installation.
all_plugin=module();all_handle=pi.install_block_fp8(all_plugin,scope='all_block128')
mlp=layer('model.layers.0.mlp.gate_proj',(129,129));mlp_method=all_plugin.Fp8LinearMethod(config())
mlp_method.create_weights(mlp,weight_loader=loader);mlp_method.process_weights_after_loading(mlp)
assert tuple(mlp._gk_block_fp8.weight.shape)==(2,129,128)
summary={'status':'PASS_CPU_STRUCTURE_ONLY','no_import_time_torch_or_patch':True,
    'class_name_keeps_loader_v2':True,'selected_upstream_loader_unchanged':True,
    'ordinary_and_unselected_stock_lifecycle':fallbacks,'no_2d_noop_view':True,
    'higher_rank_shape_restore_CPU_only':True,'prepared_owner_unchanged_after_policy_switch':True,
    'initial':initial,'mixed':mixed,'final':handle.snapshot(),'rejected':rejected,
    'all_block_selected_count':all_handle.snapshot()['selected_layer_count'],
    'device_or_model_execution':False}
(a.out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('initial','mixed','final')}))
