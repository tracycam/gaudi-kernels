"""Structural adapter test with CPU vendor TopK, never a device kernel claim."""
import argparse,json,sys,types
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
import gaudi_kernels.experimental_router_post as api
torch.manual_seed(928386);calls=[];sentinel=object()
def original(*args):calls.append(args);return sentinel
class FakeHpuMetadata:
 device=types.SimpleNamespace(type='hpu');requires_grad=False
 def __init__(self,t):self.t=t;self.shape=t.shape
 def float(self):return self.t.float()
captured=[]
def post(scores,ids,renorm,factor,*,variant):
 captured.append((scores.clone(),ids.clone(),renorm,factor,variant));w=scores.gather(1,ids)
 if renorm:w=w/w.sum(-1,keepdim=True)
 if factor!=1.:w=w*factor
 return w,ids.int()
saved=api.post_top8;api.post_top8=post
dispatch=api.make_grouped_topk_post(original,variant='vector');x=torch.randn(1,384);metadata=FakeHpuMetadata(x);bias=torch.linspace(-.5,.5,384)
positive=[]
try:
 for scoring,correction,renorm,factor in [('sigmoid',bias,True,2.5),('softmax',bias,True,1.),('sigmoid',None,False,1.)]:
  scores=x.sigmoid() if scoring=='sigmoid' else torch.softmax(x,-1);selection=scores if correction is None else scores+correction.unsqueeze(0)
  ids=torch.topk(selection,8,dim=-1,sorted=False)[1];expected=scores.gather(1,ids)
  if renorm:expected=expected/expected.sum(-1,keepdim=True)
  if factor!=1.:expected*=factor
  actual,ai=dispatch(metadata,8,renorm,1,1,scoring,factor,correction)
  assert torch.equal(ai,ids.int()) and torch.equal(actual,expected) and torch.equal(captured[-1][0],scores)
  positive.append({'scoring':scoring,'bias':correction is not None,'weights_original_scores':True,'ordered_ids_exact':True})
 args=[metadata,8,True,1,1,'sigmoid',2.5,bias]
 for index,replacement in [(0,FakeHpuMetadata(torch.empty(2,384))),(0,FakeHpuMetadata(torch.empty(1,383))),(0,x),(1,4),(3,2),(4,2),(5,'other'),(6,float('inf'))]:
  case=list(args);case[index]=replacement;assert dispatch(*case) is sentinel and calls[-1][index] is replacement
finally:api.post_top8=saved
result={'status':'CPU_ADAPTER_STRUCTURE_PASS','device_used':False,'positive':positive,'fallback_cases':len(calls),'stub_scope':'CPU structural postprocess only, not compiled-kernel qualification'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
