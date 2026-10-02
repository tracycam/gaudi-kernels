"""Real DFlash TP8 projections and graph replay; synthetic features, no TPS claim."""
import json,os,time
from datetime import timedelta
from pathlib import Path
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
import habana_frameworks.torch.distributed.hccl
import torch.distributed as dist
from dflash_reference import load_checkpoint

rank=int(os.environ['RANK']);out=Path(os.environ['PROBE_OUT'])
torch.set_num_threads(1);torch.hpu.set_device(0)
dist.init_process_group('hccl',timeout=timedelta(seconds=180));assert dist.get_world_size()==8

def sync():hc.mark_step();torch.hpu.synchronize()
def reduce_sum(x):
    hc.mark_step();dist.all_reduce(x);return x

def gather_output(x):
    hc.mark_step();parts=[torch.empty_like(x)for _ in range(8)];dist.all_gather(parts,x)
    return torch.cat(parts,dim=-1)

def metric(x,y):
    x=x.float();y=y.float();d=x-y
    return dict(max_abs=float(d.abs().max()),relative_l2=float(d.norm()/y.norm().clamp_min(1e-10)),
                rms=float(d.square().mean().sqrt()),bitwise=bool(torch.equal(x,y)))

result=dict(rank=rank,status='RUNNING',records=[],scope='Actual five-layer BF16 weights and TP8 collective, synthetic features/anchor; no target LM head, acceptance, service or TPS claim')
def save():(out/f'rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
try:
    with torch.inference_mode():
        root=Path('/data/models/MiMo-V2.6-Pro-RL/dflash');begin=time.perf_counter()
        draft=load_checkpoint(root,'hpu',tp_rank=rank,tp_size=8,reduce_sum=reduce_sum,gather_output=gather_output);sync()
        result['load_s']=time.perf_counter()-begin;result['local_parameters']=sum(p.numel()for p in draft.parameters());save()
        # Same HPU implementation unsharded: comparison diagnoses TP boundary
        # rounding only; it is not an independent quality reference.
        full=load_checkpoint(root,'hpu')if rank==0 else None
        sync();dist.barrier();sync()
        for batch,context_len in ((1,16),(1,1024),(2,1024),(3,1024)):
            torch.manual_seed(71+batch+context_len);s=draft.spec
            features=(torch.randn(batch,context_len,len(s.target_layers)*s.hidden,dtype=torch.bfloat16)*.1).to('hpu')
            anchor=(torch.randn(batch,s.hidden,dtype=torch.bfloat16)*.1).to('hpu')
            cp=torch.arange(1024-context_len,1024,dtype=torch.int32)[None,:].expand(batch,-1).contiguous().to('hpu')
            qp=torch.arange(1024,1032,dtype=torch.int32)[None,:].expand(batch,-1).contiguous().to('hpu')
            valid=torch.ones(batch,context_len,dtype=torch.bool,device='hpu')
            context=draft.project_context(features,cp);noise=draft.noise(anchor);sync()
            eager=draft(noise,qp,context,cp,valid);sync();expected=eager.cpu();assert torch.isfinite(expected).all()
            reference=None
            if rank==0:
                fc=full.project_context(features,cp);reference=full(noise,qp,fc,cp,valid);sync();reference=reference.cpu()
            dist.barrier();sync()
            stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
            with torch.hpu.graph(graph,stream=stream):value=draft(noise+0,qp+0,context,cp+0,valid+False)
            sync();graph.replay();sync();actual=value.cpu();assert torch.equal(actual,expected)
            before=[(k.cpu(),v.cpu())for k,v in context]
            qp.add_(1);sync();changed=draft(noise,qp,context,cp,valid);sync();changed=changed.cpu()
            graph.replay();sync();mutated=value.cpu();assert torch.equal(mutated,changed)
            assert all(torch.equal(t.cpu(),saved)for pair,old in zip(context,before)for t,saved in zip(pair,old))
            # Compare replicas without treating a collective as a CPU-free graph.
            gathered=[torch.empty_like(value)for _ in range(8)];dist.all_gather(gathered,value);sync()
            assert all(torch.equal(t.cpu(),mutated)for t in gathered)
            record=dict(batch=batch,context=context_len,eager_graph_bitwise=True,mutated_graph_bitwise=True,
                        committed_context_unchanged=True,all_ranks_bitwise=True)
            if rank==0:record['unsharded_bf16_difference']=metric(expected,reference)
            result['records'].append(record);save();del graph;sync()
        result['status']='COMPLETED_DFLASH_TP8_DIAGNOSTIC';save()
finally:dist.destroy_process_group()
