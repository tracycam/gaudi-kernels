"""Real HPURMSNorm forward -> production apply, with no retained norm view.

The norm class is real vLLM. The block loader class alone is a scaffold; its
selected apply body and prepared checkpoint bytes are the production path.
"""
import argparse,json,os,sys,types
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--norm-extension',required=True);p.add_argument('--block-extension',required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--original-vendor',action='store_true');a=p.parse_args()
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from vllm.config import VllmConfig,CompilationConfig,set_current_vllm_config
from vllm_gaudi.ops.hpu_layernorm import HPURMSNorm
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'));sys.path.insert(0,str(root/'benchmarks/block_fp8_framework'))
from gaudi_kernels import vllm_norm
from gaudi_kernels.block_fp8 import prepare_block_fp8
from gaudi_kernels.production_integration import install_block_fp8
from oracle import adapted_weight,quantize_block,errors

def sync():hc.mark_step();torch.hpu.synchronize()
torch.set_num_threads(4);torch.manual_seed(281624);torch.ops.load_library(str(Path(a.norm_extension).resolve()))
config=VllmConfig(compilation_config=CompilationConfig(custom_ops=['all']))
with set_current_vllm_config(config):
 resident=HPURMSNorm(6144,dtype=torch.bfloat16)
 report=vllm_norm.install_vllm_norm('vendor',model=torch.nn.Sequential(resident))
assert report['prepared']['dispatch_rebound']==1 and not resident.weight.requires_grad,report
w0=(torch.randn(6144)*.2+1).bfloat16();resident.weight.copy_(w0);resident=resident.to('hpu');sync()
class Upstream:
 def __init__(self,config):self.quant_config=config;self.block_quant=True
class Fp8LinearMethod(Upstream):
 def create_weights(self,*args,**kwargs):raise AssertionError('loader outside gate')
 def process_weights_after_loading(self,*args):raise AssertionError('fixture already prepared')
 def apply(self,*args):raise AssertionError('must use production apply')
plugin=types.ModuleType('vllm_gaudi.ops.hpu_fp8');plugin.OrigFp8LinearMethod=Upstream;plugin.Fp8LinearMethod=Fp8LinearMethod;plugin.fp8=types.SimpleNamespace(Fp8LinearMethod=Fp8LinearMethod);sys.modules[plugin.__name__]=plugin
state=install_block_fp8(plugin,extension_path=a.block_extension)
method=plugin.Fp8LinearMethod(types.SimpleNamespace(is_checkpoint_fp8_serialized=True,weight_block_size=(128,128)))
n,k=3392,6144
w=torch.frombuffer(bytearray((a.fixture/'weight.bin').read_bytes()),dtype=torch.uint8).reshape(n,k).clone().view(torch.float8_e4m3fn)
s=torch.frombuffer(bytearray((a.fixture/'scales.bin').read_bytes()),dtype=torch.float32).reshape(27,48).clone();b=torch.zeros(n);prepared=prepare_block_fp8(w,s,b)
layer=types.SimpleNamespace(prefix='model.layers.0.self_attn.qkv_proj',_gk_block_fp8_selected=True,_gk_block_fp8=prepared.to('hpu'))
torch.save({'weight':w,'scale':s,'bias':b,'gamma':w0,'prepared_bytes':prepared.weight.view(torch.uint8),'prepared_scales':prepared.scales},out/'weights.pt')
wd=adapted_weight(prepared).double().T.contiguous();qwd=[prepared.weight[g].double().T.contiguous() for g in range(48)];wsd=[prepared.scales[:,g].repeat_interleave(128)[:n].double() for g in range(48)]
def mac(norm,activation):
 x=norm.reshape(-1,k)
 if activation=='bf16':return x.double()@wd
 q,sa,_,_=quantize_block(x);y=torch.zeros((x.shape[0],n),dtype=torch.float64)
 for g in range(48):y+=(q[g].double()@qwd[g])*(sa[g].double()*wsd[g][None,:])
 return y

def norm_oracle(x0,r0,policy):
 rr=((x0+.125).bfloat16().reshape(r0.shape)+(r0*.5).bfloat16()).bfloat16()
 square=(rr*rr).double() if policy=='vendor' else rr.double().square()
 numerator=(rr*w0).double() if policy=='vendor' else rr.double()*w0.double()
 y=(numerator*torch.rsqrt(square.mean(-1,keepdim=True)+float(torch.tensor(1e-6)))).bfloat16()
 return y.reshape(x0.shape),rr

records=[]
cases=[(m,shape,block,norm) for m,block in [(1,'bf16_fp32'),(1,'decode_a8_bf16_fp32'),(512,'bf16_fp32')] for shape in ['2d','3d','mixed'] for norm in ['vendor','fp32']]
if a.original_vendor:cases=cases[:1]
with torch.inference_mode():
 for m,layout,block_policy,norm_policy in cases:
  name=f'm{m}-{layout}-{block_policy}-{norm_policy}';dest=out/name;dest.mkdir()
  xs=(m,k) if layout=='2d' else (1,m,k);rs=(m,k) if layout in ('2d','mixed') else (1,m,k)
  x0=torch.randn(xs).bfloat16();r0=torch.randn(rs).bfloat16();offset0=(torch.randn(n)*.01).bfloat16();x=x0.to('hpu');r=r0.to('hpu');offset=offset0.to('hpu');sync()
  saved={'x':x0,'residual':r0,'offset':offset0};torch.save(saved,dest/'raw.pt')
  vllm_norm.set_norm_policy(norm_policy);state.set_policy(block_policy)
  activation='per_block_fp8' if block_policy=='decode_a8_bf16_fp32' else 'bf16'
  def producer():
   xx=x+.125;rr=r*.5
   return vllm_norm._original_forward(resident,xx,rr) if a.original_vendor else resident(xx,rr)
  def chain():
   produced,_=producer()
   return method.apply(layer,produced)+offset
  # Diagnose values before capture; delete both temporary outputs. No returned
  # or retained norm/residual extends the captured producer lifetime.
  yn,rn=producer();sync();norm0=yn.cpu();residual0=rn.cpu();del yn,rn
  ey,er=norm_oracle(x0,r0,norm_policy);normerr=float((norm0.double()-ey.double()).norm()/ey.double().norm().clamp_min(1e-30))
  assert torch.equal(residual0,er) and normerr<.003,(name,normerr)
  ref=(mac(norm0,activation).bfloat16().double()+offset0.double()).bfloat16().double()
  saved.update(norm=norm0,residual_actual=residual0,norm_oracle=ey,residual_oracle=er,qkv_oracle=ref);torch.save(saved,dest/'raw.pt')
  stream,graph=torch.hpu.Stream(),torch.hpu.HPUGraph()
  with torch.hpu.graph(graph,stream=stream):y=chain()
  sync();graph.replay(asynchronous=True);sync();actual=y.cpu();initial=errors(actual.reshape(m,n),{'adapted_fp64':ref})
  assert tuple(actual.shape)==tuple(xs[:-1])+(n,) and initial['finite'] and initial['adapted_fp64']['relative_l2']<.006,(name,initial)
  pointers=y.data_ptr();x.copy_(-x0);sync();yn,rn=producer();sync();norm1=yn.cpu();residual1=rn.cpu();del yn,rn
  ey1,er1=norm_oracle(-x0,r0,norm_policy);normerr1=float((norm1.double()-ey1.double()).norm()/ey1.double().norm().clamp_min(1e-30));assert torch.equal(residual1,er1) and normerr1<.003
  ref1=(mac(norm1,activation).bfloat16().double()+offset0.double()).bfloat16().double()
  graph.replay(asynchronous=True);sync();changed=y.cpu();changeerr=errors(changed.reshape(m,n),{'adapted_fp64':ref1})
  assert pointers==y.data_ptr() and not torch.equal(actual,changed) and changeerr['finite'] and changeerr['adapted_fp64']['relative_l2']<.006,(name,changeerr)
  saved.update(actual=actual,changed=changed,changed_norm=norm1,changed_residual=residual1,changed_norm_oracle=ey1,changed_residual_oracle=er1,changed_qkv_oracle=ref1);torch.save(saved,dest/'raw.pt')
  record={'name':name,'status':'PASS','x_shape':xs,'residual_shape':rs,'norm_policy':norm_policy,'block_policy':block_policy,'norm_relative_l2':normerr,'changed_norm_relative_l2':normerr1,'residual_bitwise':True,'initial_qkv':initial,'changed_qkv':changeerr,'stable_output_address':True,'real_HPURMSNorm':True,'actual_production_apply':True,'retained_norm_or_view':False,'checked_qkv_outputs':2*actual.numel()};records.append(record);(dest/'result.json').write_text(json.dumps(record,indent=2)+'\n');(out/'result.json').write_text(json.dumps({'status':'INCOMPLETE','records':records,'installation':report},indent=2)+'\n');print(json.dumps(record),flush=True)
  del graph,y,x,r,offset
  sync()
(out/'result.json').write_text(json.dumps({'status':'PASS','records':records,'installation':report,'scope':'real norm and production apply capture boundary, minimal block loader scaffold; no model quality or timing claim'},indent=2)+'\n')
