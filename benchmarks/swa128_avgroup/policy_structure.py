"""CPU-only dispatch/owner audit. Tensor metadata stand-ins are not HPU evidence."""
import argparse
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import torch

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
events=[]
class T(torch.Tensor):
    @property
    def device(self):return torch.device('hpu')
    def item(self):raise AssertionError('No device value readback allowed')
    def cpu(self):raise AssertionError('No device readback allowed')
    def data_ptr(self):raise AssertionError('No address borrowing allowed')
def tensor(shape,dtype=torch.bfloat16):return torch.empty(shape,dtype=dtype).as_subclass(T)
def publish(name,**kwargs):
    for i in range(1,len(name.split('.'))+1):
        key='.'.join(name.split('.')[:i])
        if key not in sys.modules:
            sys.modules[key]=ModuleType(key);sys.modules[key].__path__=[]
    sys.modules[name].__dict__.update(kwargs)
def kernel(name):
    def run(*args):events.append(name);return tensor((1,16,128))
    return run
torch.ops.gaudi_swa128_avgroup.group=kernel('av_hoist')
torch.ops.gaudi_swa128.forward_window_fast_fp32=kernel('fast')
torch.ops.gaudi_swa128.reshape=lambda x,shape:x.reshape(shape)
class Cache:
    def __init__(self,name):self.name=name
    def __call__(self,new,owner,slot,**kwargs):events.append(self.name);return owner
class Impl:
    def forward(self,*args):events.append('vendor');return tensor((1,2048))
    def common_attention_args(self,blocks,k,v,size,ks,vs):
        return dict(block_list=blocks,key_cache=k,value_cache=v,block_size=size,
                    k_scales=ks,v_scales=vs,sinks=self.sinks,scale=192**-.5)
publish('vllm_gaudi.attention.backends.hpu_attn',HPUAttentionImpl=Impl,_set_fetch_by_id=lambda *a:None)
publish('vllm.v1.attention.backend',AttentionType=SimpleNamespace(DECODER='decoder'))
publish('gaudi_kernels.vllm_swa_rotary',install_rotary_boundary=lambda *a:None,
        rotary_boundary_snapshot=lambda **k:{'scope':'CPU stand-in'},prepare_rotary_boundary=lambda m:{})
# Remove the synthetic package parent so the actual implementation is imported.
sys.modules.pop('gaudi_kernels',None)
from gaudi_kernels.vllm_swa_install import install_vllm_swa,set_swa_policy,snapshot_swa_counters
impl=Impl()
for k,v in dict(sliding_window=128,num_heads=16,num_kv_heads=1,head_size=192,head_size_v=128,
    enable_fp8_attn=False,alibi_slopes=None,is_chunked_attention=False,attn_type='decoder',
    kv_sharing_target_layer_name=None,sinks=tensor((16,)),k_cache=Cache('key_write'),v_cache=Cache('value_write')).items():setattr(impl,k,v)
q,k,v=tensor((1,3072)),tensor((1,192)),tensor((1,128))
cache=(tensor((32768,1,192)),tensor((32768,1,128)),None,None)
md=SimpleNamespace(is_prompt=False,block_size=128,window_block_list=tensor((2,),torch.long),
    window_block_groups=tensor((2,),torch.long),input_positions=tensor((1,1),torch.long),
    window_block_mapping=tensor((2,1)),window_attn_bias=tensor((2,128)),slot_mapping=tensor((1,),torch.long))
install_vllm_swa('vendor');rows=[]
for policy,change,expected in [('vendor',None,['vendor']),('fp32_fast',None,['key_write','value_write','fast']),
    ('fp32_av_hoist',None,['key_write','value_write','av_hoist']),('fp32_av_hoist','prompt',['vendor']),
    ('fp32_av_hoist','full_attention',['vendor']),('fp32_av_hoist','wrong_cache',['vendor']),
    ('fp32_av_hoist','many_positions',['vendor'])]:
    set_swa_policy(policy);events.clear();md.is_prompt=change=='prompt';impl.sliding_window=256 if change=='full_attention' else 128
    md.input_positions=tensor((2,),torch.long) if change=='many_positions' else tensor((1,1),torch.long)
    chosen=list(cache) if change=='wrong_cache' else cache
    y=impl.forward(SimpleNamespace(layer_name='layer1'),q,k,v,chosen,md)
    assert events==expected,(policy,change,events);assert y.shape==(1,2048)
    rows.append({'policy':policy,'mutation':change,'events':list(events)})
snapshot=snapshot_swa_counters();assert snapshot['by_policy']['fp32_av_hoist']['eligible_custom_calls']==1
assert snapshot['by_policy']['fp32_av_hoist']['structural_fallback_calls']==4
snapshot_swa_counters(reset=True);assert all(not any(row.values())for row in snapshot_swa_counters()['by_policy'].values())
set_swa_policy('vendor');del torch.ops.gaudi_swa128_avgroup.group
try:set_swa_policy('fp32_av_hoist')
except RuntimeError:pass
else:raise AssertionError('unloaded AV-hoist accepted')
assert snapshot_swa_counters()['policy']=='vendor'
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'status':'PASS_CPU_STRUCTURE',
    'device_verified':False,'rows':rows,'snapshot':snapshot,'unloaded_policy_rejected_before_forward':True,
    'scope':'real installer/guard/dispatch code with CPU metadata/cache stand-ins; no arithmetic or native graph claim'},indent=2)+'\n')
print('PASS_CPU_STRUCTURE',len(rows))
