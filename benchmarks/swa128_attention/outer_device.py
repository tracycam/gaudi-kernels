"""Real HPUAttentionImpl, original index_copy cache writes, owned shape edges."""
import argparse,hashlib,inspect,json,os,sys
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('--extension',required=True);p.add_argument('--policy',choices=['fp32','fp32_fast'],default='fp32_fast');p.add_argument('--producer',choices=['pointwise','rotary'],default='pointwise');p.add_argument('--rope-boundary',choices=['original','owned'],default='original');a=p.parse_args()
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),VLLM_CONTIGUOUS_PA='false')
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from vllm_gaudi.extension.runtime import get_config
get_config(model_type='mimo',fp32_softmax=False,fused_block_softmax=False,fused_block_softmax_adjustment=False,per_token_kv_scaling_support=False)
from vllm_gaudi.attention.backends.hpu_attn import HPUAttentionImpl
from vllm_gaudi.ops.hpu_rotary_embedding import HPURotaryEmbedding
from oracle import full_reference
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.vllm_swa_install import install_vllm_swa,set_swa_policy,prepare_vllm_swa

def sync():hc.mark_step();torch.hpu.synchronize()
def error(x,y):
 d=x.double()-y.double();return {'finite':bool(torch.isfinite(x).all()),'relative_l2':float(d.norm()/y.double().norm().clamp_min(1e-30)),'max_abs':float(d.abs().max())}
torch.set_num_threads(4);torch.manual_seed(281962);torch.ops.load_library(str(Path(a.extension).resolve()))
original_rope=HPURotaryEmbedding.forward_oot
original=HPUAttentionImpl.forward;source=inspect.getsource(original);(out/'original-forward.py').write_text(source)
installation=install_vllm_swa();assert installation['policy']=='vendor'
records=[]
with torch.inference_mode():
 for rank in (2,3):
  dest=out/f'rank{rank}';dest.mkdir();shape=(1,3072) if rank==2 else (1,1,3072);outshape=(*shape[:-1],2048)
  q0=torch.randn(shape).bfloat16();key0=torch.randn(1,192).bfloat16();val0=torch.randn(1,128).bfloat16();kc0=torch.randn(1024,1,192).bfloat16();vc0=torch.randn(1024,1,128).bfloat16();sink0=torch.randn(16).bfloat16();offset0=(torch.randn(outshape)*.01).bfloat16()
  q,key,val,kc,vc,sinks,offset=[x.to('hpu') for x in (q0,key0,val0,kc0,vc0,sink0,offset0)]
  blocks=torch.tensor([2,7,7,7],dtype=torch.long,device='hpu');groups=torch.tensor([0,-1,-1,-1],dtype=torch.long,device='hpu');positions=torch.tensor([[127]],dtype=torch.long,device='hpu');mapping=torch.tensor([[1],[0],[0],[0]],dtype=torch.bfloat16,device='hpu');bias=torch.full((4,128),-torch.inf,dtype=torch.bfloat16,device='hpu');slot=torch.tensor([[383]],dtype=torch.long,device='hpu')
  md=SimpleNamespace(is_prompt=False,block_size=128,slot_mapping=slot,input_positions=positions,seq_lens_tensor=None,
                     block_list=blocks,block_groups=groups,block_mapping=mapping,attn_bias=bias,
                     window_block_list=blocks,window_block_groups=groups,window_block_mapping=mapping,window_attn_bias=bias,
                     chunked_block_list=None)
  impl=HPUAttentionImpl(16,192,192**-.5,1,None,128,'auto',sinks=sinks,head_size_v=128)
  cache=(kc,vc,None,None)
  if a.producer=='rotary':
   from vllm.config import VllmConfig,CompilationConfig,set_current_vllm_config
   with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=['all']))):
    rope=HPURotaryEmbedding(192,64,32768,10000,True,torch.bfloat16).to('hpu')
   packed0=torch.cat((q0.reshape(1,3072),key0,val0),dim=-1).reshape(*shape[:-1],3392);packed=packed0.to('hpu')
   prepare_vllm_swa(torch.nn.ModuleList([impl,rope]))
   if a.rope_boundary=='original':rope._forward_method=original_rope.__get__(rope,type(rope))
  sync();cache_addresses=(kc.data_ptr(),vc.data_ptr())
  torch.save({'q':q0,'key':key0,'value':val0,'key_cache':kc0,'value_cache':vc0,'sinks':sink0,'offset':offset0},dest/'inputs.pt')
  def prepare(pos,selected,active=True):
   logical=(max(0,pos-127)//128)*128;count=(pos//128)-(logical//128)+1
   ids=selected[:count]+[7]*(4-count);gg=[0]*count+[-1]*(4-count)
   if not active:gg=[-1]*4
   bb=torch.full((4,128),-torch.inf,dtype=torch.bfloat16)
   for i in range(count):
    if active:
     absolute=torch.arange(128)+logical+i*128;bb[i,(absolute>=pos-127)&(absolute<=pos)]=0
   current_slot=ids[count-1]*128+(pos%128)
   blocks.copy_(torch.tensor(ids,dtype=torch.long));groups.copy_(torch.tensor(gg,dtype=torch.long));positions.fill_(pos);slot.fill_(current_slot);mapping.copy_(torch.tensor([[int(g==0)] for g in gg],dtype=torch.bfloat16));bias.copy_(bb)
   sync()
   if a.producer=='rotary':
    pq,pk,pv=(packed+.125).split([3072,192,128],dim=-1)
    rq,rk=original_rope(rope,positions,pq,pk);rv=pv*.612;sync()
    query_cpu=rq.cpu();key_cpu=rk.cpu().reshape(1,192);value_cpu=rv.cpu().reshape(1,128)
    torch.save({'query':query_cpu,'key':key_cpu,'value':value_cpu},dest/f'rotary-p{pos}.pt')
   else:query_cpu=(q0+.125).bfloat16();key_cpu=(key0*.5).bfloat16();value_cpu=(val0-.25).bfloat16()
   kc0[current_slot]=key_cpu;vc0[current_slot]=value_cpu
   desc=ids[:count]+[-1]*(2-count) if active else [-1,-1]
   high,_=full_reference(query_cpu,kc0,vc0,desc,[logical,logical+128],pos,sink0)
   expected=(high.bfloat16().reshape(outshape).double()+offset0.double()).bfloat16()
   return expected,{'position':pos,'pages':ids,'groups':gg,'slot':current_slot,'high':high,'bias':bb}
  prepare(127,[2]);graphs={};outputs={}
  for policy in ('vendor',a.policy):
   set_swa_policy(policy);graph,stream=torch.hpu.HPUGraph(),torch.hpu.Stream()
   with torch.hpu.graph(graph,stream=stream):
    if a.producer=='rotary':
     pq,pk,pv=(packed+.125).split([3072,192,128],dim=-1);produced_q,produced_k=rope(positions,pq,pk);produced_v=pv*.612
    else:produced_q=q+.125;produced_k=key*.5;produced_v=val-.25
    y=impl.forward(None,produced_q,produced_k,produced_v,cache,md)+offset
   sync();graphs[policy]=graph;outputs[policy]=y
  for pos,pages,active in ((127,[2],True),(128,[5,2],True),(192,[1,6],True),(255,[6],True),(32767,[3],True),(128,[5,2],False)):
   expected,metadata=prepare(pos,pages,active);row={'rank':rank,'position':pos,'active':active,'variants':{}};raw={'reference':expected,**metadata}
   for policy,graph in graphs.items():
    address=outputs[policy].data_ptr();graph.replay(asynchronous=True);sync();actual=outputs[policy].cpu();e=error(actual,expected)
    assert e['finite'] and e['relative_l2']<(.003 if policy!='vendor' else .012),(rank,pos,policy,e)
    assert actual.shape==expected.shape and outputs[policy].data_ptr()==address
    assert (kc.data_ptr(),vc.data_ptr())==cache_addresses
    assert torch.equal(kc.cpu(),kc0) and torch.equal(vc.cpu(),vc0),'cache write mismatch or hidden materialization'
    row['variants'][policy]=e;raw[policy]=actual
   torch.save(raw,dest/f'p{pos}-active{int(active)}.pt');records.append(row);print(json.dumps(row),flush=True)
  # Full attention is ineligible even with a non-vendor global policy. Compare
  # the saved real forward and installed forward under the same full-page bias.
  impl.sliding_window=None;blocks.copy_(torch.tensor([0,1,7,7],dtype=torch.long));groups.copy_(torch.tensor([0,0,-1,-1],dtype=torch.long));mapping.copy_(torch.tensor([[1],[1],[0],[0]],dtype=torch.bfloat16));bb=torch.full((4,128),-torch.inf,dtype=torch.bfloat16);bb[:2]=0;bias.copy_(bb);sync()
  ref=original(impl,None,q,key,val,cache,md)+offset;sync();ref_cpu=ref.cpu();del ref
  set_swa_policy(a.policy);fg,fs=torch.hpu.HPUGraph(),torch.hpu.Stream()
  with torch.hpu.graph(fg,stream=fs):fallback=impl.forward(None,q,key,val,cache,md)+offset
  sync();fg.replay(asynchronous=True);sync();fallback_cpu=fallback.cpu();assert torch.equal(fallback_cpu,ref_cpu)
  torch.save({'reference':ref_cpu,'fallback':fallback_cpu},dest/'full-attention-fallback.pt')
  del graphs,outputs,graph,y,q,key,val,kc,vc,sinks,offset,cache,impl,md,fallback,fg;sync()
set_swa_policy('vendor')
(out/'result.json').write_text(json.dumps({'status':'PASS_REAL_HPU_ATTENTION_IMPL','policy':a.policy,'producer':a.producer,'rope_boundary':a.rope_boundary,'installation':installation,'records':records,'source_forward_sha256':hashlib.sha256(source.encode()).hexdigest(),'full_attention_fallback_bitwise':True,'cache_payload_bitwise':True,'stable_cache_and_context_addresses':True,'scope':'real outer class with vendor producers, original cache writes and vendor consumer; no full-model quality/TPS acceptance'},indent=2)+'\n')
