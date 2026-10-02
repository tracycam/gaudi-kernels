"""Prepared but unrun single-module gate; gathered input is a stand-in, not HCCL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
p=argparse.ArgumentParser();p.add_argument('--extension',type=Path,required=True);p.add_argument('--extension-sha256',required=True);a=p.parse_args()
if not os.environ.get('GAUDI_KERNELS_MODULE_ID') or os.environ.get('HABANA_VISIBLE_MODULES')!=os.environ['GAUDI_KERNELS_MODULE_ID']:
    raise RuntimeError('bounded single-module runner required')
assert hashlib.sha256(a.extension.read_bytes()).hexdigest()==a.extension_sha256
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.experimental_ag_norm import consume_gathered
torch.ops.load_library(str(a.extension.resolve()));torch.manual_seed(928129);torch.set_num_threads(2)
def sync():hc.mark_step();torch.hpu.synchronize()
result={'status':'STARTED','collective_tested':False,'scope':'single-card stand-in AG producer and public graph dependency','records':[]};raw={}
try:
    with torch.inference_mode():
        for m,h in [(2,129),(1,6144)]:
            x0=torch.randn(8*m,h)*.125;r0=torch.randn(m,h).bfloat16();w0=(torch.randn(h)*.2+1).bfloat16()
            key=f'm{m}-h{h}';raw[key]={'gathered':x0,'residual':r0,'gamma':w0};torch.save(raw,out/'fixture.pt')
            x=x0.to('hpu');r=r0.to('hpu');w=w0.to('hpu');offset=torch.tensor(.03125).to('hpu');post=torch.tensor(.125,dtype=torch.bfloat16).to('hpu');sync()
            for policy in ['vendor_boundaries','fp32_statistics']:
                stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
                with torch.hpu.graph(graph,stream=stream):
                    produced=x+offset
                    rr,yy=consume_gathered(produced,r,w,policy=policy)
                    consumed=yy+post
                sync();handles=[rr.data_ptr(),yy.data_ptr(),consumed.data_ptr()]
                for changed in [False,True]:
                    current=-x0 if changed else x0;x.copy_(current);sync()
                    for _ in range(10):graph.replay(asynchronous=True)
                    sync();ar=rr.cpu();ay=yy.cpu();ac=consumed.cpu()
                    assert handles==[rr.data_ptr(),yy.data_ptr(),consumed.data_ptr()]
                    total=torch.zeros(m,h)
                    for rank in range(8):total=total+(current[rank*m:(rank+1)*m]+.03125)
                    ref_r=(total.bfloat16()+r0).bfloat16()
                    if policy=='vendor_boundaries':square=(ref_r*ref_r).double();numerator=(ref_r*w0).double()
                    else:square=ref_r.double().square();numerator=ref_r.double()*w0.double()
                    ref_y=(numerator*torch.rsqrt(square.mean(-1,keepdim=True)+float(torch.tensor(1e-6)))).bfloat16()
                    step=torch.pow(2,torch.floor(torch.log2(ref_y.float().abs().clamp_min(2**-126)))-7)
                    ulp=float(((ay.float()-ref_y.float()).abs()/step).max())
                    raw[key][policy+str(changed)]={'residual':ar,'norm':ay,'consumer':ac,'norm_oracle':ref_y};torch.save(raw,out/'fixture.pt')
                    assert torch.equal(ar,ref_r) and torch.isfinite(ay).all() and ulp<=1.01
                    assert torch.equal(ac,(ay+torch.tensor(.125,dtype=torch.bfloat16)).bfloat16())
                    result['records'].append({'M':m,'H':h,'policy':policy,'changed':changed,'norm_max_bf16_ulp':ulp,'residual_bitwise':True,'replays':10,'logical_owner_handles_stable':True})
                x.copy_(x0);sync()
        result['status']='PASS_STANDIN_ONLY_NOT_COLLECTIVE_QUALIFIED'
except Exception as e:result.update(status='FAIL',error=repr(e));raise
finally:(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
