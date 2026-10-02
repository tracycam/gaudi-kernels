"""Real vendor 3D norm -> production method apply -> 3D QKV capture gate.

The actual production method body runs with a minimal loader-class scaffold;
this checks HPU lazy-view lifetime, not full vLLM construction or model quality.
No norm/flatten intermediate is retained past chain() during capture.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import types

p=argparse.ArgumentParser();p.add_argument('--extension',required=True);p.add_argument('--fixture',required=True,type=Path);a=p.parse_args()
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),
                  GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from habana_frameworks.torch.hpex.normalization import FusedRMSNorm
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.block_fp8 import prepare_block_fp8
from gaudi_kernels.production_integration import install_block_fp8
from oracle import reference,errors
torch.set_num_threads(4);torch.manual_seed(281337)
def sync():hc.mark_step();torch.hpu.synchronize()

class Upstream:
    def __init__(self,config):self.quant_config=config;self.block_quant=True

class Fp8LinearMethod(Upstream):
    def create_weights(self,*args,**kwargs):raise AssertionError('loader is outside this device boundary gate')
    def process_weights_after_loading(self,*args):raise AssertionError('already prepared fixture')
    def apply(self,*args):raise AssertionError('selected layer must use production apply')

plugin=types.ModuleType('vllm_gaudi.ops.hpu_fp8');plugin.OrigFp8LinearMethod=Upstream
plugin.Fp8LinearMethod=Fp8LinearMethod;plugin.fp8=types.SimpleNamespace(Fp8LinearMethod=Fp8LinearMethod)
sys.modules[plugin.__name__]=plugin
state=install_block_fp8(plugin,extension_path=a.extension)
config=types.SimpleNamespace(is_checkpoint_fp8_serialized=True,weight_block_size=(128,128))
method=plugin.Fp8LinearMethod(config)
n,k=3392,6144
w=torch.frombuffer(bytearray((a.fixture/'weight.bin').read_bytes()),dtype=torch.uint8).reshape(n,k).clone().view(torch.float8_e4m3fn)
s=torch.frombuffer(bytearray((a.fixture/'scales.bin').read_bytes()),dtype=torch.float32).reshape(27,48).clone()
b=torch.zeros(n);cpu=prepare_block_fp8(w,s,b)
layer=types.SimpleNamespace(prefix='model.layers.0.self_attn.qkv_proj',_gk_block_fp8_selected=True,_gk_block_fp8=cpu.to('hpu'))
torch.save({'weight':w,'scale':s,'bias':b,'prepared_bytes':cpu.weight.view(torch.uint8),'prepared_scales':cpu.scales},out/'weights.pt')
records=[]
with torch.inference_mode():
 for m,policy in [(1,'bf16_fp32'),(1,'decode_a8_bf16_fp32'),(2,'bf16_fp32'),(512,'bf16_fp32')]:
    name=f'm{m}-{policy}';dest=out/name;dest.mkdir()
    x0=torch.randn(1,m,k).bfloat16();r0=torch.randn(1,m,k).bfloat16();g0=(torch.randn(k)*.1+1).bfloat16()
    offset0=(torch.randn(1,1,n)*.01).bfloat16()
    x=x0.to('hpu');r=r0.to('hpu');g=g0.to('hpu');offset=offset0.to('hpu');sync()
    saved={'x':x0,'residual':r0,'gamma':g0,'post_restore_offset':offset0};torch.save(saved,dest/'raw.pt')
    # The diagnostic output is copied/deleted before capture. The captured
    # producer is newly computed and not returned or retained by chain().
    norm=FusedRMSNorm.apply(x+r,g,1e-6);sync();norm0=norm.cpu();del norm
    activation='per_block_fp8' if policy=='decode_a8_bf16_fp32' and m==1 else 'bf16'
    mac_refs=reference(norm0.reshape(m,k),w,s,b,cpu,activation,'fp32')
    def after_consumer(refs):
        return {name:(value.bfloat16().double()+offset0.reshape(1,n).double()).bfloat16().double()
                for name,value in refs.items()}
    refs=after_consumer(mac_refs)
    saved.update(norm=norm0,mac_oracles=mac_refs,oracles=refs);torch.save(saved,dest/'raw.pt')
    sync();state.set_policy(policy)
    def chain():
        produced=FusedRMSNorm.apply(x+r,g,1e-6)
        restored=method.apply(layer,produced)
        return restored+offset
    stream,graph=torch.hpu.Stream(),torch.hpu.HPUGraph()
    with torch.hpu.graph(graph,stream=stream):y=chain()
    sync();graph.replay(asynchronous=True);sync();actual=y.cpu()
    assert tuple(actual.shape)==(1,m,n)
    initial=errors(actual.reshape(m,n),refs)
    assert initial['finite'] and initial['adapted_fp64']['relative_l2']<.006,(name,initial)
    saved['actual']=actual;torch.save(saved,dest/'raw.pt')
    x.copy_(-x0);sync()
    norm=FusedRMSNorm.apply(x+r,g,1e-6);sync();changed_norm=norm.cpu();del norm
    changed_refs=after_consumer(reference(changed_norm.reshape(m,k),w,s,b,cpu,activation,'fp32'))
    graph.replay(asynchronous=True);sync();changed=y.cpu();changed_errors=errors(changed.reshape(m,n),changed_refs)
    assert changed_errors['finite'] and changed_errors['adapted_fp64']['relative_l2']<.006
    assert not torch.equal(actual,changed)
    saved.update(changed=changed,changed_norm=changed_norm,changed_oracles=changed_refs);torch.save(saved,dest/'raw.pt')
    start=time.perf_counter()
    for _ in range(10):graph.replay(asynchronous=True)
    sync();wall=(time.perf_counter()-start)*1e6/10
    record={'name':name,'status':'PASS_3D_CAPTURE','input_shape':[1,m,k],'output_shape':[1,m,n],
        'activation':activation,'producer':'vendor FusedRMSNorm(x+residual,gamma)',
        'restore_consumer':'vendor add with persistent BF16 broadcast offset',
        'retained_norm_or_flat_view':False,'actual_production_apply_body':True,
        'initial':initial,'changed':changed_errors,'checked_outputs':2*actual.numel(),
        'ten_replay_wall_us':wall,'timing_scope':'smoke wall only, not a performance comparison',
        'policy_snapshot':state.snapshot()}
    records.append(record);(dest/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    (out/'result.json').write_text(json.dumps({'status':'INCOMPLETE','records':records},indent=2)+'\n')
    print(json.dumps(record),flush=True)
    del graph,y,x,r,g,offset
    sync()
(out/'result.json').write_text(json.dumps({'status':'PASS_3D_CAPTURE','records':records,'scope':'production apply boundary; minimal loader scaffold, not full vLLM model'},indent=2)+'\n')
