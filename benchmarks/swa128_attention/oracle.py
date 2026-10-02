"""Independent two-page BF16-boundary oracle; FP64 within each MAC.

This reference preserves plugin rounding boundaries. Native reduction trees,
exp and div may differ; it is not a bitwise Gaudi vendor reference.
"""
import torch

def visible_slots(page_ids,starts,position,total_slots):
 if type(position) is not int or position<0 or position>2147483520:raise ValueError('position must be bounded nonnegative integer')
 if len(page_ids)!=2 or len(starts)!=2:raise ValueError('exactly two page descriptors required')
 if any(type(x) is not int for x in [*page_ids,*starts]):raise ValueError('integer page descriptors required')
 valid_starts=[s for p,s in zip(page_ids,starts) if p>=0]
 if len(set(valid_starts))!=len(valid_starts):raise ValueError('duplicate logical page')
 result=[]
 for p,start in zip(page_ids,starts):
  if p==-1:result.append([]);continue
  if p<0 or p>=total_slots//128 or start<0 or start%128 or start>2147483520:raise ValueError('invalid page descriptor')
  result.append([(t,p*128+t) for t in range(128) if position-127<=start+t<=position])
 return result

def reference(query,key,value,page_ids,starts,position,sinks,scale=192**-.5):
 q=(query.float()*torch.tensor(scale,dtype=torch.float32)).bfloat16().double().reshape(-1,192);heads=q.shape[0]
 slots=visible_slots(page_ids,starts,position,key.shape[0]);maxima=[];sums=[];contexts=[];logits=[];probs=[]
 for page in slots:
  score=torch.full((heads,128),-torch.inf,dtype=torch.bfloat16);v=torch.zeros((128,128),dtype=torch.float64)
  for t,slot in page:score[:,t]=(q@key[slot,0].double()).bfloat16();v[t]=value[slot,0].double()
  maximum=torch.maximum(score.amax(-1),sinks);origin=torch.where(torch.isfinite(maximum),maximum,0)
  probability=(score-origin[:,None]).bfloat16().double().exp().bfloat16()
  sums.append(probability.double().sum(-1).bfloat16());maxima.append(maximum);contexts.append((probability.double()@v).bfloat16());logits.append(score);probs.append(probability)
 maximum=torch.maximum(*maxima);origin=torch.where(torch.isfinite(maximum),maximum,0)
 adjustment=[torch.where(torch.isfinite(m),(m-origin).bfloat16().double().exp().bfloat16(),0) for m in maxima]
 adjusted=[(s*a).bfloat16() for s,a in zip(sums,adjustment)]
 denominator=(adjusted[0].double()+adjusted[1].double()).bfloat16()
 denominator=(denominator+(sinks-origin).bfloat16().double().exp().bfloat16()).bfloat16()
 contexts_scaled=[]
 for context,adj,part in zip(contexts,adjustment,adjusted):
  d=torch.maximum(denominator,part);d=torch.where(d>0,d,1)
  factor=(adj.double()/d.double()).bfloat16();contexts_scaled.append((context*factor[:,None]).bfloat16())
 output=(contexts_scaled[0].double()+contexts_scaled[1].double()).bfloat16().reshape(1,heads,128)
 return output,{'logits':logits,'probabilities':probs,'page_max':maxima,'page_sum':sums,'page_context':contexts,'page_adjustment':adjustment,'denominator':denominator,'visible_slots':slots}

def requested_bytes(page_ids,starts,position,total_slots,heads=16):
 slots=visible_slots(page_ids,starts,position,total_slots);valid=sum(map(len,slots));kv=heads*valid*(192+128)*2
 return {'valid_tokens':valid,'head_tasks':heads,'kv_payload_per_head':valid*640,'kv_requested_all_heads':kv,'unique_kv_payload':valid*640,'query_bytes':heads*192*2,'sink_bytes':heads*2,'context_bytes':heads*128*2,'metadata_requested':heads*(4+2*2*4),'scope':'ISA requested element payload, not HBM/cache transactions; every head rereads KV'}


def full_reference(query,key,value,page_ids,starts,position,sinks):
 """Complete FP64 attention; BF16 input values, no intermediate rounding.

 Returns (unrounded FP64 context, final BF16), distinct from reference().
 """
 q=query.reshape(-1,192).double();slots=visible_slots(page_ids,starts,position,key.shape[0]);physical=[slot for page in slots for _,slot in page]
 if physical:
  score=q@key[physical,0].double().T*(192**-.5)
 else:score=torch.empty((q.shape[0],0),dtype=torch.float64)
 logits=torch.cat([score,sinks.double()[:,None]],-1);maximum=logits.amax(-1,keepdim=True);origin=torch.where(torch.isfinite(maximum),maximum,0)
 probabilities=(logits-origin).exp();denominator=probabilities.sum(-1,keepdim=True);probabilities/=torch.where(denominator>0,denominator,1)
 context=probabilities[:,:len(physical)]@value[physical,0].double() if physical else torch.zeros((q.shape[0],128),dtype=torch.float64)
 context=context.reshape(1,-1,128);return context,context.bfloat16()
