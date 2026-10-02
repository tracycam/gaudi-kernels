"""Actual MiMo method, actual rank-local projections and plugin Attention.

Single module gate: constructors see TP8 dimensions and rank0, but o_proj's
distributed reduction is explicitly disabled. Actual local projection weights,
vendor/fused QKV postprocess, original cache writes and SWA all execute on HPU.
This cannot qualify distributed reduction or whole-model logits/TPS.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

parser=argparse.ArgumentParser()
parser.add_argument('--extension',required=True)
parser.add_argument('--swa-extension',required=True)
parser.add_argument('--swa-python',required=True)
args=parser.parse_args();out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),
                  GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),VLLM_CONTIGUOUS_PA='false')
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from vllm_gaudi.extension.runtime import get_config
get_config(model_type='mimo',fp32_softmax=False,fused_block_softmax=False,
           fused_block_softmax_adjustment=False,per_token_kv_scaling_support=False)
import vllm_gaudi.ops.hpu_attention
from vllm_gaudi.ops.hpu_rotary_embedding import HPURotaryEmbedding
from vllm.config import VllmConfig,CompilationConfig,CacheConfig,set_current_vllm_config
from vllm.forward_context import set_forward_context
import vllm.model_executor.models.mimo_v2 as mimo
import vllm.model_executor.layers.linear as linear
import vllm.model_executor.parameter as parameter
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
import gaudi_kernels
gaudi_kernels.__path__.append(str(Path(args.swa_python)/'gaudi_kernels'))
from gaudi_kernels.vllm_swa_install import install_vllm_swa,prepare_vllm_swa,snapshot_swa_counters
from gaudi_kernels.vllm_qkv_postprocess import (install_vllm_qkv_postprocess,prepare_vllm_qkv_postprocess,
    set_qkv_postprocess_policy,snapshot_qkv_postprocess)


def sync():hc.mark_step();torch.hpu.synchronize()


torch.set_num_threads(4);torch.manual_seed(309628);torch.set_default_dtype(torch.bfloat16)
for library in (args.extension,args.swa_extension):torch.ops.load_library(str(Path(library).resolve()))
cache_config=CacheConfig(block_size=128,gpu_memory_utilization=.5,cache_dtype='auto')
config=VllmConfig(compilation_config=CompilationConfig(custom_ops=['all']),cache_config=cache_config)
source=inspect.getsource(mimo.MiMoV2Attention);(out/'actual-mimo-class.py').write_text(source)
for module in ('vllm_swa.py','vllm_swa_install.py','vllm_swa_rotary.py'):
    shutil.copy2(Path(args.swa_python)/'gaudi_kernels'/module,out/module)
with torch.inference_mode(),set_current_vllm_config(config):
    # The actual constructor computes real TP8 local weight shapes without a
    # distributed process group. Neither construction nor qkv forward performs
    # a collective. o_proj's one collective is disabled explicitly below.
    with patch.object(mimo,'get_tensor_model_parallel_world_size',return_value=8),\
         patch.object(linear,'get_tensor_model_parallel_world_size',return_value=8),\
         patch.object(linear,'get_tensor_model_parallel_rank',return_value=0),\
         patch.object(parameter,'get_tensor_model_parallel_world_size',return_value=8),\
         patch.object(parameter,'get_tensor_model_parallel_rank',return_value=0):
        model=mimo.MiMoV2Attention(hidden_size=6144,num_heads=128,num_kv_heads=8,
            head_dim=192,v_head_dim=128,v_scale=.612,sliding_window_size=128,
            add_swa_attention_sink_bias=True,layer_id=1,rope_theta=10000,
            max_position_embeddings=32768,cache_config=cache_config,
            partial_rotary_factor=.334,prefix='gate.layers.1.self_attn')
    assert type(model.qkv_proj) is linear.QKVParallelLinear and type(model.o_proj) is linear.RowParallelLinear
    assert tuple(model.qkv_proj.weight.shape)==(3392,6144)
    assert tuple(model.o_proj.weight.shape)==(6144,2048)
    model.o_proj.reduce_results=False
    model.qkv_proj.weight.copy_((torch.randn_like(model.qkv_proj.weight.float())*.015).bfloat16())
    model.o_proj.weight.copy_((torch.randn_like(model.o_proj.weight.float())*.015).bfloat16())
    model.attention_sink_bias.copy_(torch.randn_like(model.attention_sink_bias))
    model.requires_grad_(False);model.eval();model=model.to('hpu')
    torch.save({'qkv_weight':model.qkv_proj.weight.cpu(),'output_weight':model.o_proj.weight.cpu(),
                'sinks':model.attention_sink_bias.cpu(),'actual_rotary_cache':model.rotary_emb.cos_sin_cache.cpu()},out/'parameters.pt')
    # Construct before installation to exercise the old bound rotary dispatch.
    original_rotary_fn=getattr(model.rotary_emb._forward_method,'__func__',None)
    swa_install=install_vllm_swa('fp32_fast');swa_prepare=prepare_vllm_swa(model)
    qkv_install=install_vllm_qkv_postprocess();qkv_prepare=prepare_vllm_qkv_postprocess(model)
    assert qkv_prepare['matched']==1,qkv_prepare
    kc0=torch.randn(1024,1,192).bfloat16();vc0=torch.randn(1024,1,128).bfloat16()
    kc,vc=kc0.to('hpu'),vc0.to('hpu');model.attn.kv_cache=(kc,vc,None,None)
    hidden0=torch.randn(1,6144).bfloat16();offset0=(torch.randn(1,6144)*.01).bfloat16()
    hidden,offset=hidden0.to('hpu'),offset0.to('hpu')
    blocks=torch.tensor([2,7,7,7],dtype=torch.long,device='hpu')
    groups=torch.tensor([0,-1,-1,-1],dtype=torch.long,device='hpu')
    positions=torch.tensor([[127]],dtype=torch.long,device='hpu')
    mapping=torch.tensor([[1],[0],[0],[0]],dtype=torch.bfloat16,device='hpu')
    bias=torch.full((4,128),-torch.inf,dtype=torch.bfloat16,device='hpu')
    slot=torch.tensor([[383]],dtype=torch.long,device='hpu')
    md=SimpleNamespace(is_prompt=False,block_size=128,slot_mapping=slot,input_positions=positions,
        seq_lens_tensor=None,block_list=blocks,block_groups=groups,block_mapping=mapping,attn_bias=bias,
        window_block_list=blocks,window_block_groups=groups,window_block_mapping=mapping,
        window_attn_bias=bias,chunked_block_list=None)
    sync();cache_addresses=(kc.data_ptr(),vc.data_ptr());cache_objects=(id(kc),id(vc))
    torch.save({'hidden':hidden0,'offset':offset0,'key_cache':kc0,'value_cache':vc0},out/'inputs.pt')

    def prepare(position,pages,active=True,iteration=0):
        logical=(max(0,position-127)//128)*128;count=position//128-logical//128+1
        ids=pages[:count]+[7]*(4-count);gg=([0]*count+[-1]*(4-count))if active else [-1]*4
        bb=torch.full((4,128),-torch.inf,dtype=torch.bfloat16)
        if active:
            for i in range(count):
                absolute=torch.arange(128)+logical+i*128;bb[i,(absolute>=position-127)&(absolute<=position)]=0
        current_slot=ids[count-1]*128+position%128
        blocks.copy_(torch.tensor(ids,dtype=torch.long));groups.copy_(torch.tensor(gg,dtype=torch.long))
        positions.fill_(position);slot.fill_(current_slot);bias.copy_(bb)
        mapping.copy_(torch.tensor([[int(g==0)]for g in gg],dtype=torch.bfloat16))
        changed=(hidden0.float()+iteration*.0625).bfloat16();hidden.copy_(changed);sync()
        return {'position':position,'pages':ids,'groups':gg,'slot':current_slot,'hidden':changed,'active':active}

    prepare(127,[2]);graphs={};outputs={};streams={}
    for capture in ('cold','warm'):
        for policy in ('vendor','fused'):
            set_qkv_postprocess_policy(policy);graph,stream=torch.hpu.HPUGraph(),torch.hpu.Stream()
            with set_forward_context({model.attn.layer_name:md},config),torch.hpu.graph(graph,stream=stream):
                result=model(positions,hidden+.125)+offset
            sync();name=capture+'_'+policy;graphs[name]=graph;outputs[name]=result;streams[name]=stream
    records=[]
    for index,(position,pages,active) in enumerate(((0,[2],True),(127,[2],True),(128,[5,2],True),
                                                  (255,[6],True),(32767,[3],True),(128,[5,2],False))):
        metadata=prepare(position,pages,active,index);raw={'metadata':metadata};reference=None
        for name,graph in graphs.items():
            address=outputs[name].data_ptr();graph.replay(asynchronous=True);sync()
            actual=outputs[name].cpu();keys=kc.cpu();values=vc.cpu()
            assert torch.isfinite(actual).all(),name
            if reference is None:reference=(actual,keys,values)
            mismatches=int((actual.view(torch.int16)!=reference[0].view(torch.int16)).sum())
            assert mismatches==0,(name,position,mismatches)
            assert torch.equal(keys.view(torch.int16),reference[1].view(torch.int16)),(name,'keys')
            assert torch.equal(values.view(torch.int16),reference[2].view(torch.int16)),(name,'values')
            assert outputs[name].data_ptr()==address and (kc.data_ptr(),vc.data_ptr())==cache_addresses
            assert (id(model.attn.kv_cache[0]),id(model.attn.kv_cache[1]))==cache_objects
            raw[name]=actual
        raw['key_cache']=reference[1];raw['value_cache']=reference[2]
        torch.save(raw,out/f'p{position}-active{int(active)}.pt')
        row={'position':position,'active':active,'all_four_output_bits_equal':True,
             'complete_cache_bits_equal':True,'stable_output_cache_addresses':True}
        records.append(row);print(json.dumps(row),flush=True)
    snapshot=snapshot_qkv_postprocess();assert snapshot['by_policy']['fused']['eligible_custom_calls']==2,snapshot
    assert snapshot['by_policy']['fused']['structural_fallback_calls']==0,snapshot
    post_path=out/'post_graph.json';post_files=sorted(post_path.rglob('*.json'))if post_path.is_dir()else[post_path]
    candidate_graphs=[g for p in post_files for g in json.loads(p.read_text())['graphs']
                      if any(n['guid']=='gk_qkv_post_cache_bf16_v2'for n in g['nodes'])]
    assert candidate_graphs,'model wrapper never compiled the QKV candidate'
    result={'status':'PASS_ACTUAL_MIMO_SWA_WRAPPER_LOCAL_PROJECTIONS','qkv_install':qkv_install,
            'qkv_prepare':qkv_prepare,'swa_install':swa_install,'swa_prepare':swa_prepare,
            'snapshot':snapshot,'swa_snapshot':snapshot_swa_counters(),'records':records,
            'candidate_graphs':[g['name']for g in candidate_graphs],
            'original_rotary_binding':getattr(original_rotary_fn,'__qualname__',str(original_rotary_fn)),
            'mimo_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'scope':'actual MiMoV2Attention, actual TP8-shaped rank-local QKV/row projections, actual plugin Attention/cache/SWA; o_proj cross-rank reduce explicitly disabled; not distributed or whole-model acceptance'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
