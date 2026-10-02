"""Device-resident greedy acceptance under graph replay, synthetic protocol gate."""
import json,os
from pathlib import Path
os.environ['PT_HPU_LAZY_MODE']='1'
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from protocol import greedy_accept

out=Path(os.environ['PROBE_OUT']);records=[]
def sync():hc.mark_step();torch.hpu.synchronize()
with torch.inference_mode():
    for b,k in ((1,3),(2,3),(3,3),(3,7)):
        drafts=torch.arange(10,10+b*k,dtype=torch.int32).view(b,k).to('hpu')
        targets=torch.zeros((b,k+1),dtype=torch.int32,device='hpu')
        lengths=torch.full((b,),k,dtype=torch.int32,device='hpu')
        remaining=torch.full((b,),k+1,dtype=torch.int32,device='hpu');sync()
        graph=torch.hpu.HPUGraph()
        with torch.hpu.graph(graph):
            tokens,advance=greedy_accept(drafts+0,targets+0,lengths+0,remaining+0,99)
            consumer=advance+17
        sync();d=drafts.cpu()
        for prefix in range(k+1):
            for mode in ('normal','eos','limited','zero-length','restore'):
                t=torch.cat((d.clone(),torch.full((b,1),77,dtype=torch.int32)),1)
                if prefix<k:t[:,prefix]=88
                if mode=='eos':t[:,prefix]=99
                l=torch.full((b,),0 if mode=='zero-length'else k,dtype=torch.int32)
                r=torch.arange(b,dtype=torch.int32)%2 if mode=='limited'else torch.full((b,),k+1,dtype=torch.int32)
                expected=greedy_accept(d,t,l,r,99)
                targets.copy_(t);lengths.copy_(l);remaining.copy_(r);sync();graph.replay();sync()
                got=(tokens.cpu(),advance.cpu());passed=all(torch.equal(x,y)for x,y in zip(got,expected))and torch.equal(consumer.cpu(),expected[1]+17)
                records.append(dict(B=b,K=k,prefix=prefix,state=mode,pass_=passed));assert passed,records[-1]
                torch.save(dict(drafts=d,targets=t,lengths=l,remaining=r,actual=got,expected=expected),out/f'b{b}-k{k}-p{prefix}-{mode}.pt')
        del graph,tokens,advance,consumer,drafts,targets,lengths,remaining;sync()
(out/'result.json').write_text(json.dumps(dict(status='PASS_DEVICE_GREEDY_PROTOCOL',cases=len(records),records=records,
    scope='Synthetic graph replay of acceptance only; not native full-cycle, random sampling, KV device commit, model quality, or TPS.'),indent=2)+'\n')
