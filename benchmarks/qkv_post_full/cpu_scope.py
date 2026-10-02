"""Real MiMo forward AST + installed SDK hook AST, CPU-storage metadata only."""
import argparse,ast,hashlib,json,runpy,sys
from pathlib import Path
from types import SimpleNamespace
import torch
p=argparse.ArgumentParser();p.add_argument('--mimo-source',type=Path,required=True);p.add_argument('--sdk-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2]
sys.argv=['installer_sdk_hooks.py','--sdk-source',str(a.sdk_source),'--mimo-source',str(a.mimo_source),'--output',str(a.output/'original-guards')]
base=runpy.run_path(str(root/'benchmarks/qkv_postprocess/installer_sdk_hooks.py'))
fixture=base['fixture'];state=fixture['post'].__globals__;adapter=base['adapter'];events=base['events'];setup=base['setup'];close=base['close'];observed=base['observed'];sdk=base['sdk'];tensor=fixture['tensor'];records=[]
assert adapter._ATTENTION_SCOPE=='swa128'
try:adapter.set_qkv_postprocess_policy('fused',attention_scope='all')
except ValueError:pass
else:raise AssertionError('unknown scope accepted')
change=adapter.set_qkv_postprocess_policy('fused',attention_scope='swa128_or_full');assert change['graph_invalidation_required']
hidden=tensor((1,6144));held=[]
for pos in (0,128,32767):
 model,positions,handles=setup();model.attn.impl.sliding_window=None;model.attn.layer_name='model.layers.0.self_attn.attn'
 # A distinct actual-owner cache is used for each layer; post fixture checks identity.
 model.rotary_emb.cos_sin_cache=tensor((32768,64));positions.fill_(pos)
 events.clear();observed.clear();model(positions,hidden)
 assert events==['qkv','fused_post','attention','output_projection'] and observed==['parent/rotary_emb','parent']
 assert list(sdk._name_stack)==['parent'];records.append({'case':'full_position_'+str(pos),'events':list(events),'sdk_stack_balanced':True})
 close(handles);held.append(model)
for kind in ('m2','prompt','window0','window256','unknown_hook','bad_scale','bad_rotary','bad_position_dtype'):
 model,positions,handles=setup();model.attn.impl.sliding_window=None;x=hidden
 if kind=='m2':x=tensor((2,6144));positions=tensor((2,),torch.int32)
 if kind=='prompt':state['ctx'].attn_metadata.is_prompt=True
 if kind=='window0':model.attn.impl.sliding_window=0
 if kind=='window256':model.attn.impl.sliding_window=256
 if kind=='unknown_hook':handles.append(model.rotary_emb.register_forward_hook(lambda module,inputs,out:(out[0]+2,out[1])))
 if kind=='bad_scale':model.v_scale=.5
 if kind=='bad_rotary':model.rotary_emb.is_neox_style=False
 if kind=='bad_position_dtype':positions=tensor((1,),torch.float32)
 reason=adapter._fallback_reason(model,positions,x);assert reason
 events.clear();model(positions,x)
 assert events.count('qkv')==1 and 'fused_post' not in events and list(sdk._name_stack)==['parent']
 if kind=='unknown_hook':assert bool(torch.all(model.attn.last_q==2))
 records.append({'case':kind,'reason':reason,'events':list(events)});close(handles)
model,positions,handles=setup();model.attn.impl.sliding_window=None;events.clear();observed.clear()
original=adapter.qkv_postprocess_cache
adapter.qkv_postprocess_cache=lambda *args:(_ for _ in ()).throw(RuntimeError('full_scope_injected_failure'))
try:model(positions,hidden)
except RuntimeError as error:assert str(error)=='full_scope_injected_failure'
else:raise AssertionError('failure disappeared')
finally:adapter.qkv_postprocess_cache=original;close(handles)
assert list(sdk._name_stack)==['parent'] and observed==['parent/rotary_emb','parent']
records.append({'case':'full_operator_exception','sdk_stack_balanced':True})
full=held[0];swa=fixture['model']();inventory=adapter.prepare_vllm_qkv_postprocess(torch.nn.ModuleList([full,swa]))
assert inventory['matched_by_attention_kind']=={'swa128':1,'full':1}
snapshot=adapter.snapshot_qkv_postprocess();assert snapshot['by_policy']['fused']['eligible_full_calls']>=3
assert len(snapshot['eligible_layers_by_policy']['fused']['full'])>=1
# Scope rollback itself invalidates graph caches and restores full fallback.
change=adapter.set_qkv_postprocess_policy('fused');assert change['graph_invalidation_required']
assert adapter._static_reason(full)=='attention_sliding_window'
records.append({'case':'rollback','attention_scope':adapter._ATTENTION_SCOPE})
# The source forward used in both layer constructors has no attention-window branch.
tree=ast.parse(a.mimo_source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MiMoV2Attention');fwd=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
assert not any(isinstance(n,ast.Attribute) and n.attr in ('sliding_window','sliding_window_size','is_compressed_softmax_layer') for n in ast.walk(fwd))
result={'status':'PASS_FULL_SCOPE_CPU_ONLY','device_qualified':False,'records':records,'inventory':inventory,'snapshot':snapshot,
 'mimo_source_sha256':hashlib.sha256(a.mimo_source.read_bytes()).hexdigest(),'sdk_source_sha256':hashlib.sha256(a.sdk_source.read_bytes()).hexdigest(),
 'scope':'actual MiMo forward and installed SDK AST with CPU-storage metadata; not HPU numeric/capture acceptance'}
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'full_scope_cases':len(records)}))
