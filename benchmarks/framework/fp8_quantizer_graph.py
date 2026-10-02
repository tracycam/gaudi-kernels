"""Same prepared weights, explicit table owners, full HPU Graph A/B replay."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--grid',choices=['main','stream'],default='main');a=p.parse_args()
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',
 GRAPH_VISUALIZATION_DIR=str(out/'graphs'),DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from validation import timings_agree
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.torch_linear import load_extension,prepare_separable_fp8,linear_fp8
from gaudi_kernels.fp8_quantizer import prepare_fp8_quantizer
load_extension(root/os.environ['GK_TORCH_BUILD']/'gaudi_kernels_torch.so')
counter=ctypes.CDLL(None).gaudi_kernel_launch_count;counter.restype=ctypes.c_ulonglong
torch.set_num_threads(4);torch.manual_seed(270930)
plans={v:prepare_fp8_quantizer(v).to('hpu') for v in ['lut1','lut4']}


def sync():hc.mark_step();torch.hpu.synchronize()


def error(actual,reference):
 return float((actual.double()-reference.double()).norm()/reference.double().norm().clamp_min(1e-30))


records=[];saved={}
shapes=[(1,1024,2048),(512,1024,2048),(513,1024,2048),(3,129,257)] if a.grid=='main' else [(1,8192,8192),(64,8192,8192)]
for m,n,k in shapes:
 original=(torch.randn(n,k)*64).clamp(-448,448).to(torch.float8_e4m3fn)
 scales=torch.rand(n)*.02+.01;bias=torch.randn(n)*.1;x0=torch.randn(m,k).bfloat16()
 prepared=prepare_separable_fp8(original,scales,bias)
 sa=x0.float().abs().amax(-1,keepdim=True).clamp_min(1e-10)*(1/448)
 qref=((x0.float()/sa).clamp(-448,448).to(torch.float8_e4m3fn).float()*.5).to(torch.float8_e4m3fn)
 activ=qref.double()*(sa.double()*2)
 dot=activ@prepared.weight.double().T*prepared.scale.double()
 oracle=(dot+bias.double()).bfloat16();changed_oracle=(-dot+bias.double()).bfloat16()
 key=f'{m}-{n}-{k}';saved[key]={'checkpoint':original,'scale':scales,'bias':bias,'input':x0,
  'prepared_bytes':prepared.weight.view(torch.uint8),'prepared_scales':prepared.scale,
  'quant_oracle':qref.view(torch.uint8),'scale_oracle':sa*2,'output_oracle':oracle}
 torch.save(saved,out/'inputs-outputs-oracles.pt')
 device=prepared.to('hpu');x=x0.to('hpu');xq=x0.to('hpu');sync()
 baseline=None
 for variant in ['existing','lut1','lut4']:
  if variant=='existing':quant,qs=torch.ops.gaudi_kernels._fp8_quant_fast(xq)
  else:
   operation=torch.ops.gaudi_kernels._fp8_quant_lut4 if variant=='lut4' else torch.ops.gaudi_kernels._fp8_quant_lut1
   quant,qs=operation(xq,plans[variant].table)
  sync();qbytes=quant.cpu().view(torch.uint8);qscale=qs.cpu()
  assert torch.equal(qbytes,qref.view(torch.uint8)) and torch.equal(qscale,sa*2),'public quantizer output differs'
  x.copy_(x0);sync();stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
  with torch.hpu.graph(graph,stream=stream):
   y=linear_fp8(x,device,activation='per_token_fp8',optimized=True,quantization=plans.get(variant))
  sync();graph.replay(asynchronous=True);sync();actual=y.cpu()
  if baseline is None:baseline=actual
  assert torch.equal(actual,baseline),'candidate changed full BF16 output'
  assert bool(torch.isfinite(actual).all()) and error(actual,oracle)<.001
  for _ in range(5):graph.replay(asynchronous=True)
  sync();before=counter()
  for _ in range(10):graph.replay(asynchronous=True)
  sync();launches=counter()-before;assert launches==10
  event=[];wall=[]
  for _ in range(5):
   begin,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
   start=time.perf_counter();begin.record(stream)
   for _ in range(80):graph.replay(asynchronous=True)
   sync();end.record(stream);end.synchronize();sync()
   event.append(begin.elapsed_time(end)*1000/80);wall.append((time.perf_counter()-start)*1e6/80)
  assert timings_agree(event,wall)
  pointer=y.data_ptr();x.copy_(x0.neg());sync();graph.replay(asynchronous=True);sync();changed=y.cpu()
  assert y.data_ptr()==pointer and not torch.equal(changed,actual) and error(changed,changed_oracle)<.001
  saved[key][variant]={'quant_bytes':qbytes,'activation_scale':qscale,'output':actual,'changed':changed}
  row={'M':m,'N':n,'K':k,'variant':variant,'quant_bytes_checked':qbytes.numel(),'quant_byte_mismatches':0,
   'scale_words_checked':qscale.numel(),'scale_word_mismatches':0,'full_outputs':m*n,'changed_bf16_outputs':0,
   'rounded_contract_relative_l2':error(actual,oracle),'changed_input_relative_l2':error(changed,changed_oracle),
   'stable_output_pointer':True,'launches_per_10_replays':launches,'event_us':event,'wall_us':wall,
   'median_event_us':statistics.median(event),'median_wall_us':statistics.median(wall)}
  records.append(row);torch.save(saved,out/'inputs-outputs-oracles.pt')
  (out/'result.json').write_text(json.dumps({'status':'INCOMPLETE','records':records},indent=2)+'\n')
  print(json.dumps(row),flush=True)
(out/'result.json').write_text(json.dumps({'status':'PASS','records':records,
 'scope':'public graph replay; explicit unchanged precision policy; not model quality/TPS'},indent=2)+'\n')
