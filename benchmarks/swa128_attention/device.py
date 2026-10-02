"""Real pinned flat_pa vs complete TPC SWA128; single-card graph gate and timing."""
import argparse,json,os,sys,time,hashlib,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--extension',required=True);p.add_argument('--replays',type=int,default=200);p.add_argument('--rounds',type=int,default=5);p.add_argument('--cases',default='all');p.add_argument('--diagnose',action='store_true');p.add_argument('--candidate-mode',choices=['bf16','fp32','fp32_fast'],default='bf16');a=p.parse_args();out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),VLLM_CONTIGUOUS_PA='false')
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from vllm_gaudi.extension.runtime import get_config
config=get_config(model_type='mimo',fp32_softmax=False,fused_block_softmax=False,fused_block_softmax_adjustment=False,per_token_kv_scaling_support=False)
from vllm_gaudi.attention.ops.hpu_paged_attn import HPUPagedAttention
from vllm_gaudi.extension.utils import Matmul,B2BMatmul,VLLMKVCache
from vllm_gaudi.extension import ops as plugin_ops
from oracle import reference,requested_bytes,full_reference
assert not config.fp32_softmax and not config.use_contiguous_pa
torch.set_num_threads(4);torch.manual_seed(281756);torch.ops.load_library(str(Path(a.extension).resolve()))
def sync():hc.mark_step();torch.hpu.synchronize()
def err(actual,expected):
 d=actual.double()-expected.double();return {'finite':bool(torch.isfinite(actual).all()),'relative_l2':float(d.norm()/expected.double().norm().clamp_min(1e-30)),'max_abs':float(d.abs().max()),'bit_mismatches':int((actual.view(torch.int16)!=expected.view(torch.int16)).sum())}
def full_error(actual,high):
 d=actual.double()-high;return {'relative_l2':float(d.norm()/high.norm().clamp_min(1e-30)),'max_abs':float(d.abs().max()),'rms':float(d.square().mean().sqrt())}
def timing(graph,stream):
 for _ in range(5):graph.replay(asynchronous=True)
 sync();events=[];walls=[]
 for _ in range(a.rounds):
  start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
  with torch.hpu.stream(stream):
   begin=time.perf_counter();start.record(stream)
   for _ in range(a.replays):graph.replay(asynchronous=True)
   end.record(stream);end.synchronize();walls.append((time.perf_counter()-begin)*1e6/a.replays);events.append(start.elapsed_time(end)*1000/a.replays)
 return {'event_us':events,'wall_us':walls,'median_event_us':statistics.median(events),'median_wall_us':statistics.median(walls),'scope':'complete captured attention; events on capture/replay current stream; synchronized wall; host replay supply may limit both'}
kfetch,vfetch=VLLMKVCache(),VLLMKVCache(is_v_cache=True);mmq,mma,b2b,b2a=Matmul(),Matmul(),B2BMatmul(),B2BMatmul()
records=[];source=Path(plugin_ops.__file__);(out/'plugin-ops.py').write_bytes(source.read_bytes())
with torch.inference_mode():
 for name,pos,slots,maskall,sink_mode in [('first',0,512,False,'none'),('phase64',64,512,False,'finite'),('phase127',127,512,False,'finite'),('cross128',128,512,False,'finite'),('split64',192,512,False,'none'),('allmask',128,512,True,'finite'),('allmask-none',128,512,True,'none'),('long32767',32767,32768,False,'finite')]:
  if a.cases!='all' and name not in a.cases.split(','):continue
  dest=out/name;dest.mkdir();heads=16;q0=torch.randn(1,1,heads*192).bfloat16();k0=torch.randn(slots,1,192).bfloat16();v0=torch.randn(slots,1,128).bfloat16();sink0=torch.randn(heads).bfloat16() if sink_mode=='finite' else torch.full((heads,),-torch.inf,dtype=torch.bfloat16)
  starts=[(pos//128-1)*128,(pos//128)*128];pages=[slots//128-1 if starts[0]>=0 else -1,1]
  if maskall:pages=[-1,-1]
  bias0=torch.full((2,128),-torch.inf,dtype=torch.bfloat16)
  for i,(physical,start) in enumerate(zip(pages,starts)):
   if physical>=0:
    valid=(torch.arange(128)+start>=pos-127)&(torch.arange(128)+start<=pos);bias0[i,valid]=0
  q,k,v,sink=[t.to('hpu') for t in [q0,k0,v0,sink0]];ids=torch.tensor(pages,dtype=torch.int32,device='hpu');st=torch.tensor(starts,dtype=torch.int32,device='hpu');position=torch.tensor([pos],dtype=torch.int32,device='hpu');blocklist=torch.tensor([max(0,i) for i in pages],dtype=torch.long,device='hpu');mapping=torch.ones(2,1,dtype=torch.bfloat16,device='hpu');groups=torch.zeros(2,dtype=torch.long,device='hpu');bias=bias0.to('hpu');sync()
  expected,stages=reference(q0.reshape(1,heads,192),k0,v0,pages,starts,pos,sink0);high,rounded=full_reference(q0,k0,v0,pages,starts,pos,sink0);saved={'query':q0,'key':k0,'value':v0,'sinks':sink0,'pages':pages,'starts':starts,'position':pos,'bias':bias0,'oracle':expected,'oracle_stages':stages,'full_fp64':high,'full_final_bf16':rounded};torch.save(saved,dest/'raw.pt')
  def vendor():return HPUPagedAttention.forward_decode(query=q,key_cache=k,value_cache=v,block_list=blocklist,block_mapping=mapping,block_bias=bias,block_groups=groups,block_size=128,scale=192**-.5,matmul_qk_op=mmq,matmul_av_op=mma,batch2block_matmul_op=b2b,block2batch_matmul_op=b2a,keys_fetch_func=kfetch.fetch_from_cache,values_fetch_func=vfetch.fetch_from_cache,position_bias=None,sinks=sink,k_scales=None,v_scales=None)
  def candidate():return ({'bf16':torch.ops.gaudi_swa128.forward,'fp32':torch.ops.gaudi_swa128.forward_fp32,'fp32_fast':torch.ops.gaudi_swa128.forward_fast_fp32}[a.candidate_mode])(q,k,v,ids,st,position,sink,192**-.5)
  outputs={};graphs={};streams={};record={'name':name,'candidate_mode':a.candidate_mode,'position':pos,'slots':slots,'sink_mode':sink_mode,'requested':requested_bytes(pages,starts,pos,slots),'variants':{}}
  for variant,fn in [('vendor',vendor),('candidate',candidate)]:
   stream,graph=torch.hpu.Stream(),torch.hpu.HPUGraph()
   with torch.hpu.graph(graph,stream=stream):y=fn()
   sync();graph.replay(asynchronous=True);sync();actual=y.cpu();target=rounded if variant=='candidate' and a.candidate_mode.startswith('fp32') else expected;e=err(actual,target);saved[variant]=actual;torch.save(saved,dest/'raw.pt');record['variants'][variant]={'initial':e,'initial_full_fp64':full_error(actual,high),'reference':'full_final_bf16' if variant=='candidate' and a.candidate_mode.startswith('fp32') else 'explicit_BF16_boundaries'};
   if variant=='candidate' and a.candidate_mode.startswith('fp32'):assert record['variants'][variant]['initial_full_fp64']['relative_l2']<.003
   print(json.dumps({'case':name,'variant':variant,'initial':e}),flush=True)
   assert tuple(actual.shape)==(1,heads,128) and e['finite'] and e['relative_l2']<.01,(name,variant,e)
   graphs[variant]=graph;streams[variant]=stream;outputs[variant]=y
  record['candidate_vs_vendor']=err(saved['candidate'],saved['vendor']);assert record['candidate_vs_vendor']['relative_l2']<.01
  q.copy_(-q0);sync();expected1,_=reference(-q0.reshape(1,heads,192),k0,v0,pages,starts,pos,sink0);high1,rounded1=full_reference(-q0,k0,v0,pages,starts,pos,sink0);saved['changed_oracle']=expected1
  for variant,graph in graphs.items():
   address=outputs[variant].data_ptr();graph.replay(asynchronous=True);sync();changed=outputs[variant].cpu();e=err(changed,rounded1 if variant=='candidate' and a.candidate_mode.startswith('fp32') else expected1);assert e['finite'] and e['relative_l2']<.01 and outputs[variant].data_ptr()==address;record['variants'][variant]['changed']=e;record['variants'][variant]['changed_full_fp64']=full_error(changed,high1);
   if variant=='candidate' and a.candidate_mode.startswith('fp32'):assert record['variants'][variant]['changed_full_fp64']['relative_l2']<.003
   saved['changed_'+variant]=changed
  torch.save(saved,dest/'raw.pt');q.copy_(q0);sync()
  for variant,graph in graphs.items():record['variants'][variant]['timing']=timing(graph,streams[variant])
  if name=='cross128':
   new_pages=list(reversed(pages));new_position=192
   ids.copy_(torch.tensor(new_pages,dtype=torch.int32));blocklist.copy_(torch.tensor(new_pages,dtype=torch.long));position.fill_(new_position)
   changed_bias=torch.full((2,128),-torch.inf,dtype=torch.bfloat16)
   for i,start in enumerate(starts):
    keep=(torch.arange(128)+start>=new_position-127)&(torch.arange(128)+start<=new_position);changed_bias[i,keep]=0
   bias.copy_(changed_bias);sync();metadata_ref,_=reference(q0.reshape(1,heads,192),k0,v0,new_pages,starts,new_position,sink0);metadata_high,metadata_final=full_reference(q0,k0,v0,new_pages,starts,new_position,sink0)
   record['metadata_replay']={};saved['metadata_reference']=metadata_ref;saved['metadata_full_fp64']=metadata_high;saved['metadata_pages']=new_pages;saved['metadata_position']=new_position
   for variant,graph in graphs.items():
    graph.replay(asynchronous=True);sync();actual=outputs[variant].cpu();e=err(actual,metadata_final if variant=='candidate' and a.candidate_mode.startswith('fp32') else metadata_ref);saved['metadata_'+variant]=actual;record['metadata_replay'][variant]=e;assert e['finite'] and e['relative_l2']<.01,(variant,e)
    if variant=='candidate' and a.candidate_mode.startswith('fp32'):
     e['full_fp64']=full_error(actual,metadata_high);assert e['full_fp64']['relative_l2']<.003
   torch.save(saved,dest/'raw.pt')
   ids.copy_(torch.tensor(pages,dtype=torch.int32));blocklist.copy_(torch.tensor([max(0,i) for i in pages],dtype=torch.long));position.fill_(pos);bias.copy_(bias0);sync()
  if a.diagnose:
   scaled=q*(192**-.5);sync();scaled0=scaled.cpu()
   mapped=plugin_ops.batch2block(scaled,mapping,b2b).view(2,heads,1,192).unflatten(1,(1,-1))
   fetched=kfetch.fetch_from_cache(k.unflatten(0,(-1,128)),blocklist).transpose(1,2).unflatten(1,(1,1)).transpose(-2,-1)
   score=mmq(mapped,fetched)+bias[:,None,None,None,:];sync();score_cpu=score.cpu().reshape(2,heads,128)
   maximum=torch.maximum(score.amax(-1,keepdim=True),sink.reshape(1,1,heads,1,1))
   probabilities=(score-torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))).exp();sync();prob_cpu=probabilities.cpu().reshape(2,heads,128)
   f32scaled=(q0.float()*torch.tensor(192**-.5,dtype=torch.float32)).bfloat16();bf16scaled=(q0*torch.tensor(192**-.5,dtype=torch.bfloat16)).bfloat16()
   saved['diagnostic_stages']={'scaled_query':scaled0,'scale_fp32_ref':f32scaled,'scale_bf16_ref':bf16scaled,'masked_scores':score_cpu,'probabilities':prob_cpu}
   stage_scores=torch.stack(stages['logits']);stage_prob=torch.stack(stages['probabilities']);finite=torch.isfinite(stage_scores)
   record['isolated_stage_diagnostic']={'query_fp32_scalar_bits':int((scaled0.view(torch.int16)!=f32scaled.view(torch.int16)).sum()),'query_bf16_scalar_bits':int((scaled0.view(torch.int16)!=bf16scaled.view(torch.int16)).sum()),'qk_score_bits':int((score_cpu.view(torch.int16)!=stage_scores.view(torch.int16)).sum()),'qk_relative_l2':float((score_cpu[finite].double()-stage_scores[finite].double()).norm()/stage_scores[finite].double().norm().clamp_min(1e-30)),'probability_bits':int((prob_cpu.view(torch.int16)!=stage_prob.view(torch.int16)).sum()),'probability_relative_l2':float((prob_cpu.double()-stage_prob.double()).norm()/stage_prob.double().norm().clamp_min(1e-30)),'scope':'isolated vendor operators may compile with different fusion; full-chain gate remains authoritative'}
   torch.save(saved,dest/'raw.pt');del scaled,mapped,fetched,score,maximum,probabilities
  record['status']='PASS_NUMERICS_TIMED';records.append(record);(dest/'result.json').write_text(json.dumps(record,indent=2)+'\n');(out/'result.json').write_text(json.dumps({'status':'INCOMPLETE','records':records},indent=2)+'\n');print(json.dumps(record),flush=True)
  del outputs,graphs,streams,graph,y,q,k,v,sink,ids,st,position,blocklist,mapping,groups,bias
  sync()
result={'status':'PASS_OPERATOR_GATE','candidate_mode':a.candidate_mode,'records':records,'plugin_ops_path':str(source),'plugin_ops_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'vendor_entry':'HPUPagedAttention.forward_decode -> flat_pa; Matmul/B2BMatmul/VLLMKVCache actual plugin classes','VLLM_CONTIGUOUS_PA':False,'scope':'M1 SWA128 full attention operator only; no production model or TPS acceptance'};(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
