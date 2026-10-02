"""CPU gate using the unchanged naming functions from the actual installed SDK.

Execute only their AST definitions with a recorder in place of htdebug, real
CPU Tensor/lock/deque objects, and no Habana import or device access.
"""
import argparse
import ast
from collections import deque
import functools
import hashlib
import json
from pathlib import Path
import runpy
import sys
import threading
from types import ModuleType,SimpleNamespace
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--sdk-source',type=Path,required=True)
parser.add_argument('--mimo-source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
sys.argv=['installer_structure.py','--mimo-source',str(args.mimo_source),'--output',str(args.output/'base-guards')]
fixture=runpy.run_path(str(Path(__file__).with_name('installer_structure.py')))
state=fixture['post'].__globals__;adapter=fixture['adapter'];events=fixture['events']
source=args.sdk_source.read_text();names=('_is_inference','_names_hook_already_registered','_pre_fwd_hook','_post_fwd_hook','_gen_grad_hook')
functions=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in names]
assert {node.name for node in functions}==set(names)
sdk=ModuleType('habana_frameworks.torch.core.torch_overwrites');sdk.__file__=str(args.sdk_source.resolve())
observed=[];sdk.__dict__.update(_name_stack=deque(['parent']),_module_dict={},_lock=threading.Lock(),
    htdebug=SimpleNamespace(_set_module_name=lambda name:observed.append(name)),environ={'PT_HPU_INFERENCE_MODE':'1'},Tensor=torch.Tensor)
exec(compile(ast.Module(body=functions,type_ignores=[]),sdk.__file__,'exec'),sdk.__dict__)
fixture['publish'](sdk.__name__);sys.modules[sdk.__name__]=sdk
for node in functions:
    (args.output/(node.name+'.py')).write_text(ast.get_source_segment(source,node)+'\n')


def setup(*,always=True,kwargs=False):
    model=fixture['model']();rope=model.rotary_emb
    rope.custom_name='rotary_emb';rope.names_hook=True
    handles=[rope.register_forward_pre_hook(sdk._pre_fwd_hook,with_kwargs=kwargs),
             rope.register_forward_hook(sdk._post_fwd_hook,always_call=always)]
    positions=fixture['tensor']((1,),torch.int32)
    state['current']=model;state['current_positions']=positions;state['available']=True
    state['ctx'].attn_metadata=SimpleNamespace(is_prompt=False,block_size=128)
    return model,positions,handles


def close(handles):
    for handle in handles:handle.remove()


records=[];model,positions,handles=setup();hidden=fixture['tensor']((1,6144))
assert adapter._rotary_hook_contract(model.rotary_emb)[0]=='sdk_naming_pair'
inventory=adapter.prepare_vllm_qkv_postprocess(model);assert inventory['matched']==1
events.clear();observed.clear();model(positions,hidden)
assert events==['qkv','fused_post','attention','output_projection']
assert observed==['parent/rotary_emb','parent'] and list(sdk._name_stack)==['parent']
records.append({'case':'exact_sdk_pair','events':list(events),'module_names':list(observed),'stack_balanced':True})

original=adapter.qkv_postprocess_cache
def fail(*unused):events.append('fused_raises');raise RuntimeError('intentional-TPC failure')
adapter.qkv_postprocess_cache=fail;events.clear();observed.clear()
try:model(positions,hidden)
except RuntimeError as error:assert str(error)=='intentional-TPC failure'
else:raise AssertionError('forward failure disappeared')
finally:adapter.qkv_postprocess_cache=original
assert events==['qkv','fused_raises'] and observed==['parent/rotary_emb','parent'] and list(sdk._name_stack)==['parent']
records.append({'case':'forward_exception','events':list(events),'module_names':list(observed),'stack_balanced':True,'original_error_preserved':True})
close(handles)

model,positions,handles=setup()
handles[0].remove()
@functools.wraps(sdk._pre_fwd_hook)
def impostor(*values):return sdk._pre_fwd_hook(*values)
handles[0]=model.rotary_emb.register_forward_pre_hook(impostor)
assert adapter._rotary_hook_contract(model.rotary_emb)[0]=='unsupported_module_hooks'
events.clear();model(positions,hidden)
assert events.count('qkv')==1 and 'fused_post' not in events and list(sdk._name_stack)==['parent']
records.append({'case':'same_name_wrapped_impostor','result':'original_fallback','events':list(events)});close(handles)

model,positions,handles=setup()
handles.append(model.rotary_emb.register_forward_hook(lambda module,inputs,output:(output[0]+2,output[1])))
events.clear();model(positions,hidden)
assert events.count('qkv')==1 and 'fused_post' not in events and bool(torch.all(model.attn.last_q==2))
assert list(sdk._name_stack)==['parent']
records.append({'case':'sdk_pair_plus_replacing_hook','result':'original_fallback_preserves_replacement'});close(handles)

model,positions,handles=setup()
handles.append(torch.nn.modules.module.register_module_forward_hook(lambda *unused:None))
assert adapter._rotary_hook_contract(model.rotary_emb)[0]=='unsupported_global_hooks'
global_inventory=adapter.prepare_vllm_qkv_postprocess(model)
assert len(global_inventory['global_hooks']['post'])==1
records.append({'case':'sdk_pair_plus_global','result':'reject'});close(handles)

for label,options,reason in [('not_always_call',{'always':False},'unsupported_hook_flags'),
                             ('with_kwargs',{'kwargs':True},'unsupported_hook_flags')]:
    model,positions,handles=setup(**options)
    assert adapter._rotary_hook_contract(model.rotary_emb)[0]==reason
    records.append({'case':label,'result':'reject_before_projection'});close(handles)
model,positions,handles=setup();model.rotary_emb.names_hook=False
assert adapter._rotary_hook_contract(model.rotary_emb)[0]=='unsupported_sdk_marker'
records.append({'case':'missing_sdk_marker','result':'reject_before_projection'});close(handles)
model,positions,handles=setup();handles[1].remove()
assert adapter._rotary_hook_contract(model.rotary_emb)[0]=='unsupported_module_hooks'
records.append({'case':'incomplete_pair','result':'reject_before_projection'});close(handles)

result={'status':'PASS_ACTUAL_SDK_FUNCTIONS_CPU_ONLY','cases':records,'inventory':inventory,
        'global_inventory':global_inventory,'sdk_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'base_guard_cases':22,'sdk_guard_cases':len(records),'device_verified':False,
        'substitutions':['htdebug._set_module_name records names instead of entering native debug API',
                         'private SDK globals use dedicated CPU deque/lock and inference flag; no global environment modified',
                         'exact SDK function AST is executed in an isolated module for identity matching'],
        'scope':'real installed-SDK Python function bodies; CPU metadata/dispatch fixture; no graph or device acceptance'}
(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'base_guard_cases':22,'sdk_guard_cases':len(records)}))
