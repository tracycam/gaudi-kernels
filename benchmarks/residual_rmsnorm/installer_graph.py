"""Real pinned HPURMSNorm installation and 3D operator-output capture smoke."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--extension',required=True);a=p.parse_args();out=Path(os.environ['PROBE_OUT'])
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from vllm.config import VllmConfig,CompilationConfig,set_current_vllm_config
from vllm_gaudi.ops.hpu_layernorm import HPURMSNorm,HPUGemmaRMSNorm
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.vllm_norm import install_vllm_norm,set_norm_policy

def sync():hc.mark_step();torch.hpu.synchronize()
torch.set_num_threads(4);torch.manual_seed(280940);torch.ops.load_library(str(Path(a.extension).resolve()))
config=VllmConfig(compilation_config=CompilationConfig(custom_ops=['all']))
with set_current_vllm_config(config):
 resident=HPURMSNorm(6144,dtype=torch.bfloat16)
 report=install_vllm_norm('vendor',model=torch.nn.Sequential(resident))
 new=HPURMSNorm(6144,dtype=torch.bfloat16)
assert report['prepared']['matched']==1 and report['prepared']['dispatch_rebound']==1 and not resident.weight.requires_grad and not new.weight.requires_grad,report
w0=(torch.randn(6144)*.2+1).bfloat16();resident.weight.copy_(w0);resident=resident.to('hpu');sync()
def oracle(x0,r0,vendor):
 xup=(x0+torch.tensor(.125,dtype=torch.bfloat16)).bfloat16();rup=(r0*torch.tensor(.5,dtype=torch.bfloat16)).bfloat16();rr=(xup+rup).bfloat16()
 squares=(rr*rr).double() if vendor else rr.double().square();numerator=(rr*w0).double() if vendor else rr.double()*w0.double()
 norm=(numerator*torch.rsqrt(squares.mean(-1,keepdim=True)+float(torch.tensor(1e-6)))).bfloat16();return norm,rr
records=[];saved={'gamma':w0,'installation':report}
for shape in [(1,1,6144),(1,2,6144)]:
 x0=torch.randn(shape).bfloat16();r0=torch.randn(shape).bfloat16();x=x0.to('hpu');r=r0.to('hpu');sync();key=str(shape);saved[key]={'x':x0,'residual':r0}
 for policy in ['vendor','fp32']:
  set_norm_policy(policy);stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
  with torch.hpu.graph(graph,stream=stream):
   upstream_x=x+.125;upstream_r=r*.5
   norm,residual=resident(upstream_x,upstream_r)
  sync();graph.replay(asynchronous=True);sync();ny,rr=norm.cpu(),residual.cpu();ey,er=oracle(x0,r0,policy=='vendor');saved[key][policy]={'norm':ny,'residual':rr,'norm_reference':ey,'residual_reference':er};torch.save(saved,out/'raw.pt')
  err=float((ny.double()-ey.double()).norm()/ey.double().norm());assert ny.shape==shape and rr.shape==shape and torch.equal(rr,er) and torch.isfinite(ny).all() and err<.003
  pointers=(norm.data_ptr(),residual.data_ptr());x.copy_(-x0);sync();graph.replay(asynchronous=True);sync();ny2,rr2=norm.cpu(),residual.cpu();ey2,er2=oracle(-x0,r0,policy=='vendor');saved[key][policy]['changed']=[ny2,rr2];torch.save(saved,out/'raw.pt')
  err2=float((ny2.double()-ey2.double()).norm()/ey2.double().norm());assert pointers==(norm.data_ptr(),residual.data_ptr()) and torch.equal(rr2,er2) and err2<.003
  x.copy_(x0);sync();record={'shape':shape,'policy':policy,'relative_l2_to_own_contract':err,'changed_input_relative_l2':err2,'residual_bitwise':True,'stable_addresses':True,'input_from_upstream_operator':True};records.append(record);print(json.dumps(record),flush=True)
set_norm_policy('vendor')
(out/'result.json').write_text(json.dumps({'status':'PASS','records':records,'installation':report,'scope':'real HPURMSNorm installer 3D capture smoke, not model quality/TPS'},indent=2)+'\n')
