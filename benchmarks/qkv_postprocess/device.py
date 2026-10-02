"""Actual vendor partial RoPE/V scalar contract vs direct-cache three-output TPC."""
import argparse,json,os,sys,time,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--extension',required=True);p.add_argument('--replays',type=int,default=200);p.add_argument('--rounds',type=int,default=3);a=p.parse_args()
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from vllm.config import VllmConfig,CompilationConfig,set_current_vllm_config
from vllm_gaudi.ops.hpu_rotary_embedding import HPURotaryEmbedding
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.qkv_postprocess import qkv_postprocess_cache
torch.set_num_threads(4);torch.manual_seed(291204);torch.ops.load_library(str(Path(a.extension).resolve()))
def sync():hc.mark_step();torch.hpu.synchronize()
def timing(g,stream):
 for _ in range(5):g.replay(asynchronous=True)
 sync();ev=[];wall=[]
 for _ in range(a.rounds):
  start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
  with torch.hpu.stream(stream):
   begin=time.perf_counter();start.record(stream)
   for _ in range(a.replays):g.replay(asynchronous=True)
   end.record(stream);end.synchronize();wall.append((time.perf_counter()-begin)*1e6/a.replays);ev.append(start.elapsed_time(end)*1000/a.replays)
 return {'event_us':ev,'wall_us':wall,'median_event_us':statistics.median(ev),'median_wall_us':statistics.median(wall),'scope':'complete QKV postprocess plus identical concat consumer; current capture stream events; synchronized wall; host supply may dominate'}
records=[]
with torch.inference_mode():
 with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=['all']))):
  rope=HPURotaryEmbedding(192,64,32768,10000,True,torch.bfloat16).to('hpu')
 sync();cache0=rope.cos_sin_cache.cpu();torch.save(cache0,out/'actual-layer-cache.pt')
 for m,producer in ((1,False),(1,True),(2,False),(8,False),(128,False)):
  dest=out/f'm{m}-producer{int(producer)}';dest.mkdir();x0=(torch.randn(m,3392)*8).bfloat16();bits=x0.view(torch.int16)
  special=torch.tensor([0,-32768,1,-32767,127,128,0x3f80,-16512],dtype=torch.int16)
  for offset in (0,128,3072,3200,3264):bits[:,offset:offset+8]=special
  x0[:,32:64]=x0[:,:32];x=x0.to('hpu');delta0=torch.zeros_like(x0);delta=delta0.to('hpu')
  pos0=(torch.arange(m,dtype=torch.int32)*127)%32768;positions=pos0.reshape(1,1).to('hpu') if m==1 else pos0.to('hpu');sync()
  torch.save({'qkv':x0,'delta':delta0,'positions':pos0,'value_scale':.612},dest/'inputs.pt')
  def vendor():
   produced=x+delta if producer else x;q,k,v=produced.split([3072,192,128],dim=-1);q,k=rope(positions,q,k);v=v*.612
   return torch.cat((q.reshape(m,3072),k.reshape(m,192),v.reshape(m,128)),dim=-1)
  def candidate():
   produced=x+delta if producer else x;q,k,v=qkv_postprocess_cache(produced,rope.cos_sin_cache,positions,.612)
   return torch.cat((q,k.reshape(m,192),v.reshape(m,128)),dim=-1)
  graphs={};streams={};outputs={}
  for name,fn in (('vendor',vendor),('candidate',candidate)):
   graph,stream=torch.hpu.HPUGraph(),torch.hpu.Stream()
   with torch.hpu.graph(graph,stream=stream):y=fn()
   sync();graphs[name]=graph;streams[name]=stream;outputs[name]=y
  row={'m':m,'temporary_producer':producer,'replays':[]}
  for phase,newpos in [('initial',pos0),('changed',(pos0+128)%32768),('long',torch.full_like(pos0,32767))]:
   positions.copy_(newpos.reshape(positions.shape));sync();saved={}
   for name,graph in graphs.items():
    address=outputs[name].data_ptr();graph.replay(asynchronous=True);sync();saved[name]=outputs[name].cpu();assert outputs[name].data_ptr()==address
   mismatch=int((saved['candidate'].view(torch.int16)!=saved['vendor'].view(torch.int16)).sum())
   rec={'phase':phase,'bit_mismatches':mismatch,'words':m*3392};row['replays'].append(rec);saved['positions']=newpos;torch.save(saved,dest/(phase+'.pt'));print(json.dumps({'m':m,'producer':producer,**rec}),flush=True)
   assert mismatch==0,(m,producer,phase,mismatch)
  for name,g in graphs.items():row[name+'_timing']=timing(g,streams[name])
  records.append(row);(dest/'result.json').write_text(json.dumps(row,indent=2)+'\n')
  del graphs,streams,outputs,graph,y,x,delta,positions;sync()
(out/'result.json').write_text(json.dumps({'status':'PASS_VENDOR_BITS_MULTI_OUTPUT_CAPTURE','records':records,'contract':'BF16 QKV before vendor two-BF16-product RoPE and BF16(.612) V multiply, BF16 outputs; actual layer cache used','scope':'standalone operator and real temporary producer gate; KV writes/SWA/full model are subsequent gates'},indent=2)+'\n')
