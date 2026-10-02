"""CPU-only control-flow audit with the exact saved MiMo forward AST.

Metadata tensors have CPU storage and report HPU device for guard exercise.
Projection/attention stand-ins record dispatch only: this is not an arithmetic,
native runtime, or actual tensor-device acceptance gate.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--mimo-source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
source=args.mimo_source.read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MiMoV2Attention')
forward=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
reduced=ast.Module(body=[ast.ClassDef(name=node.name,bases=[ast.Attribute(value=ast.Name(id='nn',ctx=ast.Load()),attr='Module',ctx=ast.Load())],keywords=[],body=[forward],decorator_list=[])],type_ignores=[])
ns={'torch':torch,'nn':torch.nn};exec(compile(ast.fix_missing_locations(reduced),str(args.mimo_source),'exec'),ns)
MiMoV2Attention=ns['MiMoV2Attention'];events=[]


class MetadataTensor(torch.Tensor):
    @property
    def device(self):return torch.device('hpu')


def tensor(shape,dtype=torch.bfloat16):return torch.zeros(shape,dtype=dtype).as_subclass(MetadataTensor)


class QKVParallelLinear(torch.nn.Module):
    def __init__(self):
        super().__init__();self.input_size=6144;self.output_size_per_partition=3392
        self.tp_size=8;self.gather_output=False;self.return_bias=True
        self.output_partition_sizes=[3072,192,128];self.bad_output=False
    def forward(self,x):
        events.append('qkv');self.last=tensor((x.shape[0],3391 if self.bad_output else 3392),x.dtype)
        return self.last,None


class RowParallelLinear(torch.nn.Module):
    def __init__(self):
        super().__init__();self.input_size_per_partition=2048;self.output_size_per_partition=6144
        self.tp_size=8;self.return_bias=True
    def forward(self,x):events.append('output_projection');return tensor((x.shape[0],6144),x.dtype),None


class HPURotaryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__();self.head_size=192;self.rotary_dim=64;self.is_neox_style=True
        self.cos_sin_cache=tensor((512,64));self._forward_method=self.forward_oot
    def forward_oot(self,p,q,k):events.append('rotary');return q,k
    def forward_native(self,p,q,k):events.append('rotary_native');return q,k
    def forward(self,*args):return self._forward_method(*args)


class HPUAttentionImpl(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for k,v in {'sliding_window':128,'num_heads':16,'num_kv_heads':1,'head_size':192,'head_size_v':128,
                    'enable_fp8_attn':False,'is_chunked_attention':False,'alibi_slopes':None,
                    'kv_sharing_target_layer_name':None,'attn_type':'decoder'}.items():setattr(self,k,v)


class HPUAttentionDiffKVImpl(HPUAttentionImpl):
    pass


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__();self.impl=HPUAttentionDiffKVImpl();self.use_direct_call=True
        self.query_quant=None;self.layer_name='layer0.attn'
    def forward(self,q,k,v):
        events.append('attention');self.last_q=q
        return tensor((q.shape[0],2048),q.dtype)


def publish(name,**values):
    parts=name.split('.')
    for index in range(1,len(parts)+1):
        prefix='.'.join(parts[:index])
        if prefix not in sys.modules:
            module=ModuleType(prefix);module.__path__=[];sys.modules[prefix]=module
    sys.modules[name].__dict__.update(values)


ctx=SimpleNamespace(attn_metadata=SimpleNamespace(is_prompt=False,block_size=128));available=True
publish('vllm.model_executor.models.mimo_v2',MiMoV2Attention=MiMoV2Attention)
publish('vllm_gaudi.ops.hpu_rotary_embedding',HPURotaryEmbedding=HPURotaryEmbedding)
publish('vllm.model_executor.layers.attention.attention',Attention=Attention)
publish('vllm_gaudi.attention.backends.hpu_attn',HPUAttentionImpl=HPUAttentionImpl)
publish('vllm_gaudi.v1.attention.backends.hpu_attn',HPUAttentionDiffKVImpl=HPUAttentionDiffKVImpl)
publish('vllm.model_executor.layers.linear',QKVParallelLinear=QKVParallelLinear,RowParallelLinear=RowParallelLinear)
publish('vllm.forward_context',get_forward_context=lambda:ctx,is_forward_context_available=lambda:available)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels import vllm_qkv_postprocess as adapter


def model():
    m=MiMoV2Attention()
    for k,v in {'hidden_size':6144,'num_heads':16,'num_kv_heads':1,'total_num_heads':128,'total_num_kv_heads':8,
                'head_dim':192,'v_head_dim':128,'q_size':3072,'k_size':192,'v_size':128,'v_scale':.612}.items():setattr(m,k,v)
    m.qkv_proj=QKVParallelLinear();m.o_proj=RowParallelLinear();m.rotary_emb=HPURotaryEmbedding();m.attn=Attention()
    return m


def post(qkv,cache,positions,scale):
    events.append('fused_post');assert qkv is current.qkv_proj.last
    assert cache is current.rotary_emb.cos_sin_cache and positions is current_positions and scale==.612
    return tensor((1,3072)),tensor((1,1,192)),tensor((1,1,128))


adapter.qkv_postprocess_cache=post
current=model();current_positions=tensor((1,),torch.int32);hidden=tensor((1,6144))
assert adapter._POLICY=='vendor'
adapter.install_vllm_qkv_postprocess();events.clear();current(current_positions,hidden)
assert events==['qkv','rotary','attention','output_projection']
try:adapter.set_qkv_postprocess_policy('fused')
except RuntimeError:pass
else:raise AssertionError('missing extension was not rejected before projection')
schema=torch.library.Library('gaudi_kernels','FRAGMENT')
schema.define('_qkv_post_cache_bf16_v2(Tensor qkv, Tensor cache, Tensor pos, float scale) -> (Tensor, Tensor, Tensor)')
adapter.set_qkv_postprocess_policy('fused');records=[]
for label in ('cold','warm','changed_position'):
    if label=='changed_position':current_positions.fill_(128)
    events.clear();current(current_positions,hidden)
    assert events==['qkv','fused_post','attention','output_projection']
    records.append({'case':label,'events':list(events)})

cases=[('m2','hidden_shape'),('float32','hidden_dtype'),('position_dtype','positions_dtype'),
       ('full_attention','attention_sliding_window'),('value_none','value_scale'),('value_different','value_scale'),
       ('cache_float32','rotary_cache_dtype'),('non_neox','rotary_geometry'),('native_rotary','rotary_dispatch'),
       ('prompt','prompt'),('metadata_missing','layer_metadata_absent'),('context_absent','forward_context_absent')]
for label,reason in cases:
    current=model();current_positions=tensor((1,),torch.int32);x=hidden
    ctx.attn_metadata=SimpleNamespace(is_prompt=False,block_size=128);available=True
    if label=='m2':x=tensor((2,6144));current_positions=tensor((2,),torch.int32)
    elif label=='float32':x=tensor((1,6144),torch.float32)
    elif label=='position_dtype':current_positions=tensor((1,),torch.float32)
    elif label=='full_attention':current.attn.impl.sliding_window=None
    elif label=='value_none':current.v_scale=None
    elif label=='value_different':current.v_scale=.5
    elif label=='cache_float32':current.rotary_emb.cos_sin_cache=tensor((512,64),torch.float32)
    elif label=='non_neox':current.rotary_emb.is_neox_style=False
    elif label=='native_rotary':current.rotary_emb._forward_method=current.rotary_emb.forward_native
    elif label=='prompt':ctx.attn_metadata.is_prompt=True
    elif label=='metadata_missing':ctx.attn_metadata={}
    elif label=='context_absent':available=False
    events.clear();assert adapter._fallback_reason(current,current_positions,x)==reason
    current(current_positions,x)
    assert events.count('qkv')==1 and 'fused_post' not in events and events[-1]=='output_projection'
    records.append({'case':label,'reason':reason,'events':list(events)})
available=True;ctx.attn_metadata=SimpleNamespace(is_prompt=False,block_size=128)
current=model();current_positions=tensor((1,),torch.int32);current.qkv_proj.bad_output=True;events.clear()
try:current(current_positions,hidden)
except RuntimeError as error:assert 'output contract' in str(error)
else:raise AssertionError('invalid projection output accepted')
assert events==['qkv'];records.append({'case':'invalid_projection_output','events':list(events),'result':'raises without fallback'})
current=model();full=model();full.attn.impl.sliding_window=None
inventory=adapter.prepare_vllm_qkv_postprocess(torch.nn.ModuleList([current,full]))
assert inventory['matched']==1 and inventory['rejected']['1']=='attention_sliding_window'
class UnknownImpl(HPUAttentionImpl):pass
current.attn.impl=UnknownImpl()
assert adapter._static_reason(current)=='attention_impl_class'
current.attn.impl=HPUAttentionDiffKVImpl()
HPUAttentionDiffKVImpl.forward=lambda self,*args:None
assert adapter._static_reason(current)=='diffkv_forward_override'
del HPUAttentionDiffKVImpl.forward
records.extend([{'case':'unknown_impl_subclass','result':'fallback'},
                {'case':'diffkv_forward_override','result':'fallback'}])
for kind in ('forward','pre','global_forward','global_pre'):
    current=model();current_positions=tensor((1,),torch.int32)
    def hook(module,*unused):
        if module is not current.rotary_emb:return None
        events.append('rotary_hook')
        if len(unused)==2:
            q,k=unused[1];return q+2,k
        p,q,k=unused[0];return p,q+2,k
    if kind=='forward':handle=current.rotary_emb.register_forward_hook(hook)
    elif kind=='pre':handle=current.rotary_emb.register_forward_pre_hook(hook)
    elif kind=='global_forward':handle=torch.nn.modules.module.register_module_forward_hook(hook)
    else:handle=torch.nn.modules.module.register_module_forward_pre_hook(hook)
    try:
        events.clear();assert adapter._fallback_reason(current,current_positions,hidden)=='rotary_forward_hooks'
        current(current_positions,hidden)
        assert events.count('qkv')==1 and events.count('rotary_hook')==1 and 'fused_post' not in events
        assert bool(torch.all(current.attn.last_q==2)),'hook replacement was discarded'
        records.append({'case':kind+'_hook','result':'fallback_before_projection','events':list(events)})
    finally:handle.remove()
result={'status':'PASS_CPU_STRUCTURE_ONLY','records':records,'inventory':inventory,'snapshot':adapter.snapshot_qkv_postprocess(),
        'actual_mimo_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'scope':'actual saved MiMo forward AST; CPU-storage metadata fixture and dispatch counters only. No device or arithmetic acceptance.'}
(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
(args.output/'actual-forward.py').write_text(ast.unparse(forward)+'\n')
print(json.dumps({'status':result['status'],'cases':len(records),'matched':inventory['matched']}))
