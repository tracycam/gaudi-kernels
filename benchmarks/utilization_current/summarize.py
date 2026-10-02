"""Read-only saved-result report; no device, model execution, or acceptance edits."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--canonical',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();base=a.canonical;refs={}
def read(path):
 raw=(base/path).read_bytes();refs[path]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def bound(t,r,e,c):
 active=min(e,t*r);return min(active*math.ceil(t/c),active+(t*r-active)//c)
def rates(t,us,active,requested):
 params=6144*512+256*6144;weight=params*17//32;f=2*t*8*params
 return dict(useful_FLOPs=f,effective_TFLOPS=f/us/1e6,unique_active_weight_scale_bytes=active*weight,logical_requested_weight_scale_bytes=requested*weight,unique_weight_TBps=active*weight/us/1e6,requested_weight_TBps=requested*weight/us/1e6,physical_HBM_TBps=None)
r=read('artifacts/builds/production-integration/model-runs/production-native-lanes-70-g/result.json');assert r['status']=='PASS'and r['candidate_accepted']and r['fp32_quality']['passed']
native={}
for label,arms in(('control',(0,3)),('full_post',(1,2))):
 runs=[v for v in r['runs']if any(v['name'].endswith('-native-policy-abba-'+str(i))for i in arms)];assert len(runs)==2
 native[label]=dict(TPS=sum(v['serving_timing']['steady_steps']for v in runs)/sum(v['serving_timing']['steady_steps']/v['serving_timing']['tps']for v in runs),arms=[v['serving_timing']for v in runs],scope='native TP8 B1,4096 prompt,160 generated; pooled128-token warmed windows; all stalls retained')
batch=[]
for n in(1,2,8):
 for label,arms in(('broadcast',(0,3)),('compact',(1,2))):
  rows=[v for v in r['batch_runs']if v['batch']==n and v['arm']in arms];assert len(rows)==2
  for v in rows:
   d=v['full_batch_decode_timing'];indices=d['selected_step_indices'];assert indices==list(range(indices[0],indices[-1]+1));s=[v['steps'][i]for i in indices]
   assert abs((s[-1]['end_ns']-s[0]['begin_ns'])/1e9-d['duration_s'])<1e-9
   assert sum(v['new_tokens']for v in s)==d['new_tokens'] and d['qualified']
  dt=sum(v['full_batch_decode_timing']['duration_s']for v in rows);nt=sum(v['full_batch_decode_timing']['new_tokens']for v in rows);pt=sum(v['prefill_completion_timing']['prompt_tokens']for v in rows);pd=sum(v['prefill_completion_timing']['duration_s']for v in rows)
  batch.append(dict(B=n,policy=label,backend='bridge',aggregate_decode_TPS=nt/dt,per_request_equivalent_TPS=nt/dt/n,selected_steps_each=[v['full_batch_decode_timing']['steps']for v in rows],prompt_tokens_each=512,prompt_completion_TPS=pt/pd,mean_completion_s=pd/2,early_decode_tokens_each=[v['prefill_completion_timing']['early_decode_tokens']for v in rows],all_corresponding_tokens_match=all(v['cross_policy_token_match']and v['repeat_match']for v in rows)))
fp8=[]
for row in read('artifacts/builds/block-fp8-native-lanes/module6-a/results/native-lanes-abba-summary.json'):
 m=int(row['name'].rsplit('m',1)[1]);us=row['candidate_us'];N,K=3392,6144;weight=N*K;scales=math.ceil(N/128)*(K//128)*4;F=2*m*N*K
 fp8.append(dict(M=m,N=N,K=K,activation='BF16',scale_math='FP32',whole_public_recipe_event_us=us,effective_TFLOPS=F/us/1e6,nominal_BF16_MME_peak_ratio_percent=100*F/us/1e6/432,original_weight_bytes=weight,original_scale_bytes=scales,weight_only_TBps=weight/us/1e6,weight_scale_TBps=(weight+scales)/us/1e6,weight_only_vs_nominal_HBM_percent=100*weight/us/1e6/2.45,physical_HBM_measured=False))
fp8.insert(0,dict(M=1,native_lanes_event_us=None,reason='native-lanes ABBA has no M1 case; production M1 blockA8 quantizer/FP8MME/Neumaier is a different recipe. Do not substitute historical20.29us native builder or A16 timing.'))
# Small-M rate uses the deployed complete graph, not experimental combine timing.
small_path='artifacts/builds/moe-combine-preload/sealed-device-a/results/combine-full-chain-a'
small=read(small_path+'/result.json');assert small['numeric_complete']and small['timing_complete'];moe_small=[]
import torch
for t in(1,2,8):
 file=base/small_path/f'm{t}/inputs.pt';refs[str(file.relative_to(base))]=hashlib.sha256(file.read_bytes()).hexdigest();fixture=torch.load(file,weights_only=False,map_location='cpu')['initial'];active=int(torch.topk(fixture['logits']+.125,8,dim=-1).indices.unique().numel())
 row=next(v for v in small['summary']if v['rows']==t and v['state']=='initial'and v['variant']=='deployed');us=row['event_median_us']
 moe_small.append(dict(T=t,E_resident=24,E_active=active,event_us=us,wall_us=row['wall_median_us'],arithmetic='deployed GP/gate/down289/ordered FP32 combine',**rates(t,us,active,t*8),TPC_MAC_peak_utilization=None,scope='whole public producer/topk/MoE/consumer; immutable synthetic E24 weights, not full-model layer timing'))
wide=[]
paths={'checkpoint':'artifacts/builds/route-wide/sealed-device-a/results/checkpoint-wide-t512-a/result.json','all384':'artifacts/builds/route-wide/sealed-device-d/results/checkpoint-wide-all384-t512-d/result.json'}
all_result=read(paths['all384']);counts_by={'checkpoint':all_result['expert_counts']['checkpoint_routes'],'all384':all_result['expert_counts']['original']}
for kind,path in paths.items():
 d=all_result if kind=='all384'else read(path);assert d['numeric_complete']and d['timing_complete'] and not d['candidate_accepted']
 counts=counts_by[kind];active=sum(n>0 for n in counts);t=512;assert sum(counts)==t*8
 for variant in('broadcast','c32','c64','c128'):
  times=[v for v in d['timing']if v['variant']==variant];us=statistics.median(v['event_us']for v in times)
  c=None if variant=='broadcast'else int(variant[1:]);tiles=t*8 if c is None else sum(math.ceil(n/c)for n in counts);cap=None if c is None else bound(t,8,384,c)
  entry=dict(distribution=kind,T=t,E_resident=384,E_active=active,mean_rows_per_active_expert=t*8/active,variant=variant,event_median_us=us,wall_median_us=statistics.median(v['wall_us']for v in times),timing_arms=len(times),**rates(t,us,active,tiles),model_qualified=False)
  entry['whole_recipe_vs_nominal_BF16_MME_percent']=entry['effective_TFLOPS']/432*100 if c else None
  if c:
   entry.update(C=c,capacity=cap,active_tiles=tiles,empty_tiles=cap-tiles,active_tile_valid_row_fraction=t*8/(tiles*c),static_valid_row_fraction=t*8/(cap*c),logical_decoder_store_bytes=cap*2*(6144*512+256*6144),empty_decoder_zero_store_bytes=(cap-tiles)*2*(6144*512+256*6144),declared_MME_FLOPs_including_padding=2*cap*c*(6144*512+256*6144))
  wide.append(entry)
# Tail case uses the saved, deliberately appended token rather than invented
# uniform counts.  It is a separate run; do not pair its time with case A.
tail_path='artifacts/builds/route-wide/sealed-device-b/results/checkpoint-wide-t513-b'
tail=read(tail_path+'/result.json');assert tail['numeric_complete']and tail['timing_complete']
tail_file=base/tail_path/'inputs.pt';refs[str(tail_file.relative_to(base))]=hashlib.sha256(tail_file.read_bytes()).hexdigest()
tail_fixture=torch.load(tail_file,weights_only=False,map_location='cpu')['original']
tail_ids=tail_fixture['ids'];assert tuple(tail_ids.shape)==(513,8)
counts=torch.bincount(tail_ids.flatten().long(),minlength=384).tolist();active=sum(n>0 for n in counts)
for variant in('broadcast','c32','c64','c128'):
 times=[v for v in tail['timing']if v['variant']==variant];us=statistics.median(v['event_us']for v in times)
 c=None if variant=='broadcast'else int(variant[1:]);tiles=513*8 if c is None else sum(math.ceil(n/c)for n in counts)
 wide.append(dict(distribution='checkpoint_append_last',T=513,E_resident=384,E_active=active,mean_rows_per_active_expert=513*8/active,variant=variant,event_median_us=us,wall_median_us=statistics.median(v['wall_us']for v in times),timing_arms=len(times),**rates(513,us,active,tiles),model_qualified=False))
direct=read('artifacts/builds/route-wide/sealed-device-c/results/checkpoint-wide-direct-t512-c/result.json')
direct_ratios={}
for variant in('c64','c128'):
 values=[]
 for trial in range(6):
  arms=[v for v in direct['timing']if v['pair']==variant and v['trial']==trial];assert len(arms)==4
  values.append(statistics.mean(v['event_us']for v in arms if v['variant']==variant)/statistics.mean(v['event_us']for v in arms if v['variant']=='c32'))
 direct_ratios[variant]=dict(six_ABBA_ratios=values,min_ratio=min(values),max_ratio=max(values))
coverage=[]
for t in(1,2,8,32,48,128,512,513):
 active=min(384,t*8);coverage.append(dict(T=t,max_E_active=active,full384_possible=active==384,mean_Me_at_max_active=t*8/active,mean_Me_if_divided_by384=t*8/384,meaning='T*8/384 is not an active-expert mean when fewer than384 experts can be selected'))
# Analytical only: this intentionally does not read the running 128K result.
# Geometry is the parent's explicit run configuration.  Frozen source proves
# O consumes local Q_heads*V_dim and the SWA mask includes exactly window_size
# keys including the current key (triu diagonal=shift-window_size+1).
frozen='artifacts/builds/production-integration/model-runs/production-native-lanes-70-g/source/production-runtime/'
for file in ('vllm/vllm/model_executor/models/mimo_v2.py','plugin/vllm_gaudi/v1/worker/hpu_model_runner.py'):
 raw=(base/frozen/file).read_bytes();refs[frozen+file]=hashlib.sha256(raw).hexdigest()
context=131072;chunk=512;chunks=context//chunk;params=6144*512+256*6144;weight=params*17//32
flops=dict(qkv_70_layers=70*2*context*3392*6144,
 o_projection_70_layers=70*2*context*(16*128)*6144,
 expert_GP_down_69_layers=69*2*context*8*params,
 full_attention_10_layers=10*2*16*(192+128)*(context*(context+1)//2),
 SWA128_60_layers=60*2*16*(192+128)*(128*context-128*127//2))
analytic=dict(status='ANALYTICAL_ONLY_NOT_DEVICE_RESULT',context_tokens=context,chunk_tokens=chunk,chunks=chunks,TP=8,
 geometry_source='parent-provided 128K run configuration, crosschecked against frozen70g MiMo model and SWA mask source; not a128K completion claim',
 full_attention_layers=10,SWA_layers=60,Q_heads_per_rank=16,KV_heads_per_rank=1,QK_dim=192,V_dim=128,
 uniform_all384_Me_per_chunk=chunk*8/384,per_rank_useful_FLOPs=flops,per_rank_subtotal_FLOPs=sum(flops.values()),TP8_subtotal_FLOPs=8*sum(flops.values()),
 excluded_from_subtotal=['first-layer dense FFN','router matmul and topk','norm/RoPE/scaling/softmax/sink','LM head','quantization/decoding','padding or masked-pair execution','communication and copies'],
 per_rank_logical_weight_bytes=dict(MoE_broadcast_69layers_per_chunk=69*chunk*8*weight,MoE_broadcast_69layers_all_chunks=69*context*8*weight,
  MoE_all384_once_each_69layers_per_chunk=69*384*weight,MoE_all384_once_each_69layers_all_chunks=69*384*weight*chunks,
  QKV_original_weight_70layers_per_chunk=70*3392*6144,QKV_original_weight_70layers_all_chunks=70*3392*6144*chunks),
 limitations=['Broadcast per-route payload is algorithmic load demand; caches and physical transactions are unmeasured.',
  'All384 once-per-expert volume assumes every expert active in every chunk and perfect within-chunk reuse; it is not a measured route-tile traffic result.',
  'SWA count is the useful128-key causal window; actual prefill may compute masked/padded entries.',
  'QKV byte budget excludes scale payload and O/first-MLP/router weights; it is not total model bytes.',
  'These are useful arithmetic subtotals, not executed instruction counts, full-model utilization, timing lower bounds, or TPS predictions.'])
report=dict(scope='Read-only archival rates. No new device work or model acceptance. Operator rates are TP-local;70g service rates are full TP8.',source_sha256=refs,model70g=dict(status=r['status'],candidate_accepted=r['candidate_accepted'],config=r['config'],native_B1=native,batch=batch,long_context_native=next(v['serving_timing']for v in r['long_context']['runs']if v['name']=='native'),hardware_BW_and_FLOPS_utilization=None,hardware_utilization_reason='No full-model physical byte counter or precision-aware complete operation-count denominator established.'),FP8_QKV=fp8,MXFP4_small=moe_small,MXFP4_large=wide,wide_direct_ABBA=direct_ratios,expert_coverage=coverage,prefill128K_analytical=analytic,denominators=dict(nominal_HBM_TBps=2.45,nominal_BF16_MME_TFLOPS=432,nominal_FP8_MME_TFLOPS=865,source='existing docs/UTILIZATION-20260927.md nominal figures; not measured clocks, physical traffic or engine busy cycles',one_expert_weight_parameters=6144*512+256*6144,one_expert_MXFP4_weight_scale_bytes=(6144*512+256*6144)*17//32),limits=['Current route wrappers T<=513; model70g scheduler max_num_batched_tokens512.','128K total context is not one128K MoE invocation; if chunk512 remains, uniform expert Me stays10.667.','Native-lanes route uses BF16 MME despite FP8 stored weights.','Logical request or unique-byte throughput must not be labelled BMON physical bandwidth.','The new all384 case substitutes IDs but retains real checkpoint weights/x/routing; it is not natural model routing or model quality.','FP32 effectiveFLOPs use useful active routes, exclude padding/gate/scales/metadata; comparison to nominal MME peak is whole-recipe efficiency, not measured MME utilization.'])
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(model70g='PASS',FP8_rows=len(fp8),MoE_rows=len(moe_small)+len(wide),sources=len(refs))))
