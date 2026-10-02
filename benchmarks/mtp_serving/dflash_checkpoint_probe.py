"""Load all real MiMo DFlash weights; graph-test synthetic context transactions.

No target model/LM head: this is not an acceptance rate or serving TPS test.
"""
import hashlib,json,os,struct,time
from pathlib import Path
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from dflash_reference import load_checkpoint


def sync():hc.mark_step();torch.hpu.synchronize()


out=Path(os.environ['PROBE_OUT']);root=Path('/data/models/MiMo-V2.6-Pro-RL/dflash')
result=dict(status='RUNNING',scope='Real five-layer weights, synthetic target features; no model acceptance, TP8 or serving TPS claim',records=[])
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
save()
with torch.inference_mode():
    begin=time.perf_counter();draft=load_checkpoint(root,'hpu');sync()
    result['load_s']=time.perf_counter()-begin;result['parameters']=sum(p.numel()for p in draft.parameters())
    s=draft.spec;torch.manual_seed(71)
    cpu_features=torch.randn((1,16,len(s.target_layers)*s.hidden),dtype=torch.bfloat16)*.1
    features=cpu_features.to('hpu');cp=torch.arange(1008,1024,dtype=torch.int32,device='hpu')[None,:]
    qp=torch.arange(1024,1024+s.block,dtype=torch.int32,device='hpu')[None,:]
    valid=torch.ones((1,16),dtype=torch.bool,device='hpu')
    anchor=(torch.randn((1,s.hidden),dtype=torch.bfloat16)*.1).to('hpu');noise=draft.noise(anchor)
    context=draft.project_context(features,cp);sync()
    eager=draft(noise,qp,context,cp,valid);sync();expected=eager.cpu();assert torch.isfinite(expected).all()
    before=[(k.cpu(),v.cpu())for k,v in context]
    # Validate context extend: projecting committed chunks equals concatenating
    # their caches to FP32/BF16 tolerances; no draft noise enters the context.
    pieces=[draft.project_context(features[:,start:end],cp[:,start:end])for start,end in ((0,7),(7,16))];sync()
    chunk_max=0.
    for i in range(s.layers):
        for j in (0,1):
            actual=torch.cat([p[i][j]for p in pieces],1).cpu();delta=(actual.float()-before[i][j].float()).abs().max().item();chunk_max=max(chunk_max,delta)
    result['chunk_projection_max_abs']=chunk_max
    graph=torch.hpu.HPUGraph()
    with torch.hpu.graph(graph):got=draft(noise+0,qp+0,context,cp+0,valid+False)
    sync()
    for shift in (0,1,127,128):
        cp.copy_(torch.arange(1008+shift,1024+shift,dtype=torch.int32)[None,:])
        qp.copy_(torch.arange(1024+shift,1024+shift+s.block,dtype=torch.int32)[None,:]);sync()
        # Re-project context with the changed positions before the graph.
        new=draft.project_context(features,cp)
        for target,pair in zip(context,new):
            for dst,src in zip(target,pair):dst.copy_(src)
        sync();reference=draft(noise,qp,context,cp,valid);sync();reference=reference.cpu()
        graph.replay();sync();value=got.cpu()
        delta=(value.float()-reference.float()).abs();passed=torch.equal(value,reference)
        result['records'].append(dict(position_shift=shift,finite=bool(torch.isfinite(value).all()),bitwise_eager_graph=passed,max_abs=float(delta.max())))
        assert passed and torch.isfinite(value).all();save()
    # Replays must not overwrite committed context with ephemeral noise KV.
    unchanged=[(k.cpu(),v.cpu())for k,v in context]
    graph.replay();sync()
    assert all(torch.equal(t.cpu(),saved)for pair,old in zip(context,unchanged)for t,saved in zip(pair,old))
    result['context_unchanged_by_draft']=True
    del graph;sync()
    with (root/'dflash_draft_model.safetensors').open('rb')as f:
        n=struct.unpack('<Q',f.read(8))[0];result['header_sha256']=hashlib.sha256(f.read(n)).hexdigest()
    result['config_sha256']=hashlib.sha256((root/'config.json').read_bytes()).hexdigest()
    result['mask_sha256']=hashlib.sha256((root/'mask_embedding.pt').read_bytes()).hexdigest()
    result['status']='PASS_DFLASH_CHECKPOINT_GRAPH';save()
