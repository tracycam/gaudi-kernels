"""Prepared/unrun gate: one vendor TopK feeds original and both public consumers."""
import argparse,hashlib,json,os
from pathlib import Path
import sys
p=argparse.ArgumentParser();p.add_argument('--extension',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
if not os.environ.get('GAUDI_KERNELS_MODULE_ID') or os.environ.get('HABANA_VISIBLE_MODULES')!=os.environ['GAUDI_KERNELS_MODULE_ID']:raise RuntimeError('bounded single-module runner required')
assert hashlib.sha256(a.extension.read_bytes()).hexdigest()==a.sha256
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir();os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',PT_HPU_ENABLE_DIV_PRECISE='1',DUMP_POST_GRAPHS=str(out/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.experimental_router_post import post_top8
torch.ops.load_library(str(a.extension.resolve()));torch.manual_seed(928385);torch.set_num_threads(2)
def sync():hc.mark_step();torch.hpu.synchronize()
def compare(actual,expected):
 assert torch.equal(torch.isnan(actual),torch.isnan(expected)) and torch.equal(torch.isinf(actual),torch.isinf(expected))
 assert torch.equal(torch.signbit(actual[~torch.isnan(expected)]),torch.signbit(expected[~torch.isnan(expected)]))
 finite=torch.isfinite(expected)
 if not finite.any():return 0
 def ordered(x):
  bits=x.contiguous().view(torch.int32).to(torch.int64)&0xffffffff
  return torch.where(bits&0x80000000!=0,0x80000000-(bits&0x7fffffff),0x80000000+bits)
 error=int((ordered(actual[finite])-ordered(expected[finite])).abs().max());assert error<=1,error;return error
raw={};result={'status':'STARTED','PT_HPU_ENABLE_DIV_PRECISE':1,'selection':'one shared vendor TopK per graph, sorted=False; exact ordered IDs required','records':[]}
try:
 with torch.inference_mode():
  cases=[('sigmoid',True,2.5),('softmax',True,1.),('ties',True,2.5),('near_threshold',True,1.),('no_renorm',False,1.),('tiny',False,1.),('tiny_scaled',False,1.+2**-30),('zero',True,2.5)]
  for name,renorm,factor in cases:
   cpu=torch.randn(1,384);bias0=torch.randn(384)*.1
   if name=='ties':cpu.zero_();bias0.zero_()
   if name=='near_threshold':cpu=torch.linspace(-1e-6,1e-6,384).reshape(1,384);bias0.zero_()
   if name.startswith('tiny'):cpu.fill_(1e-40)
   if name=='zero':cpu.zero_()
   raw[name]={'input':cpu,'bias':bias0,'renormalize':renorm,'factor':factor};torch.save(raw,out/'fixtures.pt')
   x=cpu.to('hpu');bias=bias0.to('hpu');zero=torch.tensor(0.).to('hpu');coeff=torch.arange(1,9,dtype=torch.float32).reshape(1,8).to('hpu');sync()
   stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
   with torch.hpu.graph(graph,stream=stream):
    # Arithmetic-producing inputs exercise the real public bridge dependency.
    # Tiny raw-input guard fixtures must not first flush values through ADD 0.
    produced=x if name.startswith('tiny') else x+zero
    scores=produced if name.startswith('tiny') or name=='zero' else (torch.softmax(produced,-1) if name=='softmax' else produced.sigmoid())
    ids=torch.topk(scores+bias.unsqueeze(0),8,dim=-1,sorted=False)[1]
    gathered=scores.gather(1,ids);denominator=gathered.sum(-1,keepdim=True)
    baseline=gathered/denominator if renorm else gathered
    if factor!=1.:baseline=baseline*factor
    baseline_ids=ids.to(torch.int32)
    scalar=post_top8(scores,ids,renorm,factor,variant='scalar',diagnostics=True)
    vector=post_top8(scores,ids,renorm,factor,variant='vector',diagnostics=True)
    consumed=[(v*coeff).sum(-1) for v in (baseline,scalar[0],vector[0])]
   sync();owners=[v.data_ptr() for v in (baseline,baseline_ids,*scalar,*vector,*consumed)]
   for changed in (False,True):
    new=cpu.roll(31,-1) if changed else cpu;new_bias=bias0.roll(47) if changed else bias0
    x.copy_(new);bias.copy_(new_bias);sync()
    for _ in range(10):graph.replay(asynchronous=True)
    sync();reference=baseline.cpu();ordered_ids=baseline_ids.cpu();den=denominator.cpu()
    actuals=[]
    for output in (scalar,vector):
     weight,index,total=[v.cpu() for v in output];error=compare(weight,reference)
     assert torch.equal(index,ordered_ids) and torch.equal(total.view(torch.int32),den.view(torch.int32))
     actuals.append({'weights':weight,'ids':index,'sum':total,'max_vendor_ulp':error})
    assert owners==[v.data_ptr() for v in (baseline,baseline_ids,*scalar,*vector,*consumed)]
    consumer_values=[v.cpu() for v in consumed]
    consumer_ulp=[compare(v,consumer_values[0]) for v in consumer_values[1:]]
    raw[name][str(changed)]={'baseline':reference,'ids':ordered_ids,'sum':den,'candidate':actuals,'consumer':consumer_values};torch.save(raw,out/'fixtures.pt')
    result['records'].append({'case':name,'changed':changed,'ordered_ids_exact':True,'denominator_bits_exact':True,'max_vendor_ulp':[v['max_vendor_ulp'] for v in actuals],'consumer_ulp':consumer_ulp,'ten_replays':True,'logical_owner_handles_stable':True})
  result['status']='PASS_PUBLIC_POST_TOPK_GATE_NO_PERFORMANCE_CLAIM'
except Exception as e:result.update(status='FAIL',error=repr(e));raise
finally:(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
