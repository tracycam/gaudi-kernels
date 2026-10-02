"""CPU actual-decoder-AST protocol gate. No HPU arithmetic emulation."""
import argparse,ast,json,runpy,sys
from pathlib import Path
from types import SimpleNamespace,ModuleType
import torch
p=argparse.ArgumentParser();p.add_argument('--mimo-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
saved=sys.argv;sys.argv=['installer_structure.py','--mimo-source',str(a.mimo_source),'--output',str(a.output/'post-fixture')]
fixture=runpy.run_path(str(root/'benchmarks/qkv_postprocess/installer_structure.py'));sys.argv=saved
from gaudi_kernels import vllm_norm_grid24 as grid,norm_grid24_binding as binding
from gaudi_kernels import vllm_norm as norm,production_integration as block
events=fixture['events'];tensor=fixture['tensor'];publish=fixture['publish'];post=fixture['adapter']
source=ast.parse(a.mimo_source.read_text());cls=next(n for n in source.body if isinstance(n,ast.ClassDef)and n.name=='MiMoV2FlashDecoderLayer')
forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='forward')
tree=ast.Module(body=[ast.ClassDef(name=cls.name,bases=[ast.Attribute(ast.Name('torch',ast.Load()),'nn',ast.Load())],keywords=[],body=[forward],decorator_list=[])],type_ignores=[])
# Replace the AST base with Module; retain the exact saved forward body.
tree.body[0].bases=[ast.Attribute(ast.Attribute(ast.Name('torch',ast.Load()),'nn',ast.Load()),'Module',ast.Load())]
scope={'torch':torch};exec(compile(ast.fix_missing_locations(tree),str(a.mimo_source),'exec'),scope);Decoder=scope[cls.name]
publish('vllm.model_executor.models.mimo_v2',MiMoV2FlashDecoderLayer=Decoder)
sys.modules['vllm.forward_context'].is_forward_context_available=lambda:True
sys.modules['vllm.forward_context'].get_forward_context=lambda:SimpleNamespace(attn_metadata=SimpleNamespace(is_prompt=False,block_size=128))
class Norm(torch.nn.Module):
 def __init__(self):super().__init__();self.weight=tensor((6144,));self.variance_epsilon=1e-6;self.variance_size_override=None;self._forward_method=self.forward_oot
 def forward_oot(self,x,residual=None):events.append('ordinary_norm');return x if residual is None else(x,residual)
 def forward(self,x,residual=None):return self._forward_method(x,residual)
class PostNorm(torch.nn.Module):
 def forward(self,x,r):events.append('postnorm');return x,r
class MLP(torch.nn.Module):
 def forward(self,x):events.append('mlp');return x
class Method:pass
state=SimpleNamespace(method_class=Method,policy='decode_a8_bf16_fp32',audit_enabled=False,audit_frame=None,audit_contract='fp32_arithmetic_v1',python_apply_branch_counts={},audit_records=[],reduction_policy='neumaier_isa_fp32',reducer=lambda:None)
from collections import Counter
state.python_apply_branch_counts=Counter();block._active=state;norm._target=Norm;norm._forward=Norm.forward_oot;norm._policy='fp32'
def model():
 m=Decoder();m.layer_id=1;m.input_layernorm=Norm();m.post_attention_layernorm=PostNorm();m.mlp=MLP();m.self_attn=fixture['model']()
 q=m.self_attn.qkv_proj;q._gk_block_fp8_policy_owner=state;q._gk_block_fp8_selected=True;q._gk_block_fp8=SimpleNamespace(n=3392,k=6144);q.quant_method=Method();q.bias=None;q.prefix='model.layers.1.self_attn.qkv_proj';return m
def producer(x,r,g,e):events.append('grid24');return r,tensor((48,1,128),torch.float8_e4m3fn),tensor((48,1,1),torch.float32)
torch.ops.gaudi_norm_grid24.block128=producer
def consume(q,s,p,**kwargs):events.append('mm_reduce');return tensor((1,3392))
grid.linear_block_fp8_quantized=consume
def postprocess(qkv,cache,positions,scale):
 events.append('fused_post');return tensor((1,3072)),tensor((1,1,192)),tensor((1,1,128))
post.qkv_postprocess_cache=postprocess
binding.verify=lambda **kwargs:{'guid':'gk_norm_block128_grid24_v1','cpu_fixture':True}
post.set_qkv_postprocess_policy('fused')
m=model();grid.install('grid24',model=m);x=tensor((1,6144));r=tensor((1,6144));pos=tensor((1,),torch.long)
events.clear();m(pos,x,r)
assert events==['grid24','mm_reduce','fused_post','attention','output_projection','postnorm','mlp'],events
records=[dict(case='explicit_chain',events=list(events))]
for name,setup,args,undo in [
 ('M2',lambda:None,(pos,tensor((2,6144)),tensor((2,6144))),lambda:None),
 ('residual_none',lambda:None,(pos,x,None),lambda:None),
 ('A16',lambda:setattr(state,'policy','bf16_fp32'),(pos,x,r),lambda:setattr(state,'policy','decode_a8_bf16_fp32')),
 ('CPU_reference',lambda:setattr(state,'policy','cpu_native_a8_fp32_v1'),(pos,x,r),lambda:setattr(state,'policy','decode_a8_bf16_fp32')),
 ('vendor_norm',lambda:setattr(norm,'_policy','vendor'),(pos,x,r),lambda:setattr(norm,'_policy','fp32')),
 ('legacy_audit',lambda:(setattr(state,'audit_enabled',True),setattr(state,'audit_contract','legacy')),(pos,x,r),lambda:(setattr(state,'audit_enabled',False),setattr(state,'audit_contract','fp32_arithmetic_v1'))),
]:
 setup();events.clear();m(*args);undo();assert 'grid24'not in events and events.count('qkv')==1,(name,events);records.append(dict(case=name,events=list(events)))
for name,module in [('norm',m.input_layernorm),('qkv',m.self_attn.qkv_proj),('attention',m.self_attn)]:
 hook=module.register_forward_hook(lambda *args:events.append('unknown_hook'))
 events.clear();m(pos,x,r);hook.remove();assert 'grid24'not in events and 'unknown_hook'in events;records.append(dict(case=name+'_hook',events=list(events)))
# Exact SDK hooks keep their nesting; outer decoder hooks are called normally.
sdk=ModuleType('habana_frameworks.torch.core.torch_overwrites');stack=[]
def before(module,args):stack.append(module.custom_name);events.append(module.custom_name+'.begin')
def after(module,args,output):assert stack.pop()==module.custom_name;events.append(module.custom_name+'.end')
sdk._pre_fwd_hook=before;sdk._post_fwd_hook=after;sys.modules[sdk.__name__]=sdk
handles=[]
for name,module in [('norm',m.input_layernorm),('qkv',m.self_attn.qkv_proj),('self_attn',m.self_attn)]:
 module.names_hook=True;module.custom_name=name;handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after,always_call=True)])
outer=[m.register_forward_pre_hook(lambda *args:events.append('decoder.begin')),m.register_forward_hook(lambda *args:events.append('decoder.end'))]
events.clear();m(pos,x,r);assert not stack
assert events==['decoder.begin','norm.begin','grid24','norm.end','self_attn.begin','qkv.begin','mm_reduce','qkv.end','fused_post','attention','output_projection','self_attn.end','postnorm','mlp','decoder.end'],events
records.append(dict(case='sdk_and_outer_hooks',events=list(events)))
old=grid.linear_block_fp8_quantized
def fail(*args,**kwargs):raise RuntimeError('injected MME failure')
grid.linear_block_fp8_quantized=fail
try:m(pos,x,r)
except RuntimeError as e:assert str(e)=='injected MME failure'
else:raise AssertionError('failure swallowed')
assert not stack;grid.linear_block_fp8_quantized=old
for h in handles+outer:h.remove()
# An active v1 frame must call the actual-grid audit hook exactly once.
state.audit_enabled=True;state.audit_frame={'positions':[128]}
original_audit=grid.audit.record
def audit_record(*args):events.append('actual_grid_audit');state.audit_records.append({'actual_grid_fixture':True})
grid.audit.record=audit_record;events.clear();m(pos,x,r);grid.audit.record=original_audit
assert events.count('actual_grid_audit')==1 and grid.snapshot()['counts']['actual_grid_audit_calls']==1
records.append(dict(case='audit_dispatch',events=list(events)))
# Full-post opt-in must not inherit into the separately qualified grid24 scope.
post.set_qkv_postprocess_policy('fused',attention_scope='swa128_or_full')
m.self_attn.attn.impl.sliding_window=None
assert post._static_reason(m.self_attn) is None
assert grid._static(m)=='post_attention_sliding_window'
events.clear();m(pos,x,r)
assert 'grid24' not in events and 'fused_post' in events and events.count('qkv')==1
records.append(dict(case='full_post_does_not_expand_grid24',events=list(events)))
m.self_attn.attn.impl.sliding_window=128
post.set_qkv_postprocess_policy('fused')
report=dict(status='PASS_CPU_STRUCTURE_ONLY',records=records,snapshot=grid.snapshot(),scope='actual saved decoder forward AST; CPU-storage metadata tensors and operation stand-ins; no device/model acceptance')
(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'decoder-forward.py').write_text(ast.unparse(forward)+'\n');print(json.dumps({'status':report['status'],'cases':len(records)}))
