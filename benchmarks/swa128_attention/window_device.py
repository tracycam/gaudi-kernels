"""Explicit production flat_pa adapter, original Long metadata, changed capture inputs."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--extension',required=True);p.add_argument('--candidate-policy',choices=['fp32','fp32_fast'],default='fp32');a=p.parse_args();out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),VLLM_CONTIGUOUS_PA='false')
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from vllm_gaudi.extension.runtime import get_config
get_config(model_type='mimo',fp32_softmax=False,fused_block_softmax=False,fused_block_softmax_adjustment=False,per_token_kv_scaling_support=False)
from vllm_gaudi.extension import ops
from vllm_gaudi.extension.utils import Matmul,B2BMatmul,VLLMKVCache
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.vllm_swa import flat_pa_window,eligible_window_call
from oracle import full_reference

def sync():hc.mark_step();torch.hpu.synchronize()
def errors(a,b):
 d=a.double()-b.double();return {'relative_l2':float(d.norm()/b.double().norm().clamp_min(1e-30)),'max_abs':float(d.abs().max()),'finite':bool(torch.isfinite(a).all())}
torch.set_num_threads(4);torch.manual_seed(281848);torch.ops.load_library(str(Path(a.extension).resolve()))
q0=torch.randn(1,1,3072).bfloat16();k0=torch.randn(1024,1,192).bfloat16();v0=torch.randn(1024,1,128).bfloat16();sink0=torch.randn(16).bfloat16();offset0=(torch.randn(1,16,128)*.01).bfloat16()
q,k,v,sink,offset=[t.to('hpu') for t in [q0,k0,v0,sink0,offset0]]
blocks=torch.tensor([2,7,7,7],dtype=torch.long,device='hpu');groups=torch.tensor([0,-1,-1,-1],dtype=torch.long,device='hpu');positions=torch.tensor([[127]],dtype=torch.long,device='hpu');mapping=torch.tensor([[1],[0],[0],[0]],dtype=torch.bfloat16,device='hpu');bias=torch.full((4,128),-torch.inf,dtype=torch.bfloat16,device='hpu')
kf,vf=VLLMKVCache(),VLLMKVCache(is_v_cache=True);mmq,mma,b2b,b2a=Matmul(),Matmul(),B2BMatmul(),B2BMatmul();sync()
saved={'query':q0,'key':k0,'value':v0,'sinks':sink0,'offset':offset0};torch.save(saved,out/'inputs.pt')
def kwargs(query):return dict(query=query,key_cache=k,value_cache=v,block_list=blocks,block_groups=groups,block_mapping=mapping,block_bias=bias,block_size=128,scale=192**-.5,matmul_qk_op=mmq,matmul_av_op=mma,batch2block_matmul_op=b2b,block2batch_matmul_op=b2a,keys_fetch_func=kf.fetch_from_cache,values_fetch_func=vf.fetch_from_cache,position_bias=None,sinks=sink,k_scales=None,v_scales=None)
def prepare(pos,selected,active=True):
 logical=(max(0,pos-127)//128)*128;count=(pos//128)-(logical//128)+1
 ids=selected[:count]+[7]*(4-count);gg=[0]*count+[-1]*(4-count)
 if not active:gg=[-1]*4
 bb=torch.full((4,128),-torch.inf,dtype=torch.bfloat16)
 for i in range(count):
  if active:
   absolute=torch.arange(128)+logical+i*128;bb[i,(absolute>=pos-127)&(absolute<=pos)]=0
 blocks.copy_(torch.tensor(ids,dtype=torch.long));groups.copy_(torch.tensor(gg,dtype=torch.long));positions.fill_(pos);mapping.copy_(torch.tensor([[1 if g==0 else 0] for g in gg],dtype=torch.bfloat16));bias.copy_(bb);sync()
 desc=ids[:count]+[-1]*(2-count) if active else [-1,-1];starts=[logical,logical+128]
 high,_=full_reference((q0+.125).bfloat16(),k0,v0,desc,starts,pos,sink0)
 return (high.bfloat16().double()+offset0.double()).bfloat16(),{'position':pos,'blocks':ids,'groups':gg,'starts':starts,'bias':bb,'full_fp64_before_consumer':high}
records=[];outputs={};graphs={}
with torch.inference_mode():
 expected,meta=prepare(127,[2])
 for policy in ['vendor',a.candidate_policy]:
  def chain():
   produced=q+.125
   return flat_pa_window(window_block_list=blocks,window_block_groups=groups,input_positions=positions,sliding_window=128,policy=policy,**kwargs(produced))+offset
  assert eligible_window_call(kwargs(q),blocks,groups,positions,128)
  graph,stream=torch.hpu.HPUGraph(),torch.hpu.Stream()
  with torch.hpu.graph(graph,stream=stream):y=chain()
  sync();graphs[policy]=graph;outputs[policy]=y
 for pos,selected,active in [(127,[2],True),(128,[5,2],True),(192,[1,6],True),(255,[6],True),(32767,[3],True),(128,[5,2],False)]:
  expected,meta=prepare(pos,selected,active);row={'position':pos,'active':active,'metadata':{k:v for k,v in meta.items() if not isinstance(v,torch.Tensor)},'variants':{}};raw={'reference':expected,**meta}
  for policy,graph in graphs.items():
   ptr=outputs[policy].data_ptr();graph.replay(asynchronous=True);sync();actual=outputs[policy].cpu();e=errors(actual,expected);assert e['finite'] and e['relative_l2']<(.003 if policy!='vendor' else .012) and ptr==outputs[policy].data_ptr(),(policy,pos,e);row['variants'][policy]=e;raw[policy]=actual
  name=f'p{pos}-active{int(active)}';torch.save(raw,out/(name+'.pt'));records.append(row);print(json.dumps(row),flush=True)
 # Full-attention fallback: both physical pages contribute all256 tokens. The
 # same adapter must preserve the actual vendor output exactly when not SWA.
 blocks.copy_(torch.tensor([0,1,7,7],dtype=torch.long));groups.copy_(torch.tensor([0,0,-1,-1],dtype=torch.long));mapping.copy_(torch.tensor([[1],[1],[0],[0]],dtype=torch.bfloat16));full_bias=torch.full((4,128),-torch.inf,dtype=torch.bfloat16);full_bias[:2]=0;bias.copy_(full_bias);sync()
 assert not eligible_window_call(kwargs(q),blocks,groups,positions,None)
 reference=ops.flat_pa(**kwargs(q));sync();reference_cpu=reference.cpu();del reference
 fallback_graph=torch.hpu.HPUGraph();fallback_stream=torch.hpu.Stream()
 with torch.hpu.graph(fallback_graph,stream=fallback_stream):fallback=flat_pa_window(window_block_list=blocks,window_block_groups=groups,input_positions=positions,sliding_window=None,policy=a.candidate_policy,**kwargs(q))
 sync();fallback_graph.replay(asynchronous=True);sync();fallback_cpu=fallback.cpu();assert torch.equal(fallback_cpu,reference_cpu);torch.save({'actual':fallback_cpu,'vendor':reference_cpu,'bias':full_bias},out/'full-attention-fallback.pt')
result={'status':'PASS_EXPLICIT_WINDOW_ADAPTER','candidate_policy':a.candidate_policy,'records':records,'metadata_torch_dtype':'int64','input_positions_shape':[1,1],'full_attention_fallback_bitwise':True,'producer':'vendor add creates temporary Q, no temporary reshape','consumer':'vendor add on BF16 context','scope':'actual flat_pa adapter and dynamic production-format tensors; no HPUAttentionImpl outer-wrapper installer or model readiness'};(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
