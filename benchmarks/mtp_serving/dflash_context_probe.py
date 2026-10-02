"""Graph-replayed ragged DFlash commit/reset against an independent CPU ring."""
import json,os
from pathlib import Path
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from dflash_context import CommittedContext
out=Path(os.environ['PROBE_OUT']);records=[]
def sync():hc.mark_step();torch.hpu.synchronize()
with torch.inference_mode():
    for batch in (1,2,3):
        ring=CommittedContext(layers=2,batch=batch,window=1024,kv_heads=1,dim=8,device='hpu')
        positions=torch.zeros(batch,8,dtype=torch.int32,device='hpu');counts=torch.zeros(batch,dtype=torch.int32,device='hpu')
        projected=tuple(tuple(torch.zeros(batch,8,1,8,dtype=torch.bfloat16,device='hpu')for _ in range(2))for _ in range(2))
        resets=torch.zeros(batch,dtype=torch.bool,device='hpu');sync();stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
        with torch.hpu.graph(graph,stream=stream):
            ring.reset(resets+False);ring.commit(tuple(tuple(x+0 for x in pair)for pair in projected),positions+0,counts+0)
            kv,cp,valid=ring.state();consumer=sum((k.float()+v.float()).sum()for k,v in kv)+valid.int().sum()
        sync();expected_pos=torch.full((batch,1024),-1,dtype=torch.int32)
        expected=[torch.zeros(batch,1024,1,8,dtype=torch.bfloat16)for _ in range(4)]
        for step in range(24):
            pos=torch.arange(1016+step*8,1024+step*8)[None,:].expand(batch,-1).int()
            count=torch.tensor([(step+3*b)%9 for b in range(batch)],dtype=torch.int32)
            reset=torch.tensor([step in (11,19)and b==step%batch for b in range(batch)])
            data=[(torch.arange(batch*8*8).reshape(batch,8,1,8)+step+i*10).bfloat16()for i in range(4)]
            positions.copy_(pos);counts.copy_(count);resets.copy_(reset)
            for dst,src in zip([x for pair in projected for x in pair],data):dst.copy_(src)
            sync();graph.replay();sync();cp=ring.state()[1].cpu();actual=[x.cpu()for pair in ring.state()[0]for x in pair]
            for b in range(batch):
                if reset[b]:expected_pos[b].fill_(-1)
                for j in range(int(count[b])):
                    slot=int(pos[b,j])%1024;expected_pos[b,slot]=pos[b,j]
                    for target,src in zip(expected,data):target[b,slot]=src[b,j]
            assert torch.equal(cp,expected_pos)
            assert all(torch.equal(x,y)for x,y in zip(actual,expected))
            want=sum(x.float().sum()for x in expected)+(expected_pos>=0).int().sum()
            assert torch.equal(consumer.cpu(),want)
            records.append(dict(batch=batch,step=step,counts=count.tolist(),reset=reset.tolist(),state_bitwise=True,consumer_bitwise=True))
        del graph;sync()
(out/'result.json').write_text(json.dumps(dict(status='PASS_DFLASH_COMMITTED_CONTEXT_GRAPH',records=records,
    scope='Synthetic KV, graph replay through scatter and subsequent consumer; not actual model acceptance'),indent=2)+'\n')
