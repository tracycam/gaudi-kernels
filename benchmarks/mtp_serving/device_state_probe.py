"""Graph-replay gate for SWA commit and the DSpark Markov sampling component.

Synthetic tensors only. Does not claim a DSpark checkpoint or serving backend.
"""
import json
import os
from pathlib import Path
os.environ['PT_HPU_LAZY_MODE']='1'
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from device_state import commit_ring,dspark_greedy,visibility

def sync():hc.mark_step();torch.hpu.synchronize()

out=Path(os.environ['PROBE_OUT']);records=[]
with torch.inference_mode():
    b,w,t=3,128,8
    ring=torch.zeros((b,w,2),dtype=torch.float32,device='hpu')
    tags=torch.full((b,w),-1,dtype=torch.int32,device='hpu')
    lengths=torch.zeros(b,dtype=torch.int32,device='hpu')
    advance=torch.ones(b,dtype=torch.int32,device='hpu')
    columns=torch.arange(t,dtype=torch.int32,device='hpu')
    rows=torch.arange(b,dtype=torch.int32,device='hpu')
    sync();graph=torch.hpu.HPUGraph()
    with torch.hpu.graph(graph):
        # Device producer uses the committed cursor on each replay.
        staging=torch.stack(((lengths[:,None]+columns).float(),rows[:,None].expand(b,t).float()),-1)
        mask=visibility(tags+0,lengths+0,t,w)
        updated,next_tags,next_lengths=commit_ring(ring+0,tags+0,lengths+0,staging,advance+0)
        ring.copy_(updated);tags.copy_(next_tags);lengths.copy_(next_lengths)
    sync()
    for start in (0,1,126,127,128,129,255,4095,4096):
        for advances in ((0,1,8),(8,3,0),(1,1,1)):
            cpu_ring=torch.full((b,w,2),-999,dtype=torch.float32)
            cpu_tags=torch.full((b,w),-1,dtype=torch.int32)
            cpu_lengths=torch.tensor([start,start+2,start+4],dtype=torch.int32)
            for row,n in enumerate(cpu_lengths.tolist()):
                for p in range(max(0,n-w),n):
                    cpu_tags[row,p%w]=p;cpu_ring[row,p%w]=torch.tensor([p,row])
            cpu_advance=torch.tensor(advances,dtype=torch.int32)
            ring.copy_(cpu_ring);tags.copy_(cpu_tags);lengths.copy_(cpu_lengths);advance.copy_(cpu_advance);sync()
            for replay in range(3):
                staged=torch.stack(((cpu_lengths[:,None]+torch.arange(t)).float(),torch.arange(b)[:,None].expand(b,t).float()),-1)
                expected_mask=visibility(cpu_tags,cpu_lengths,t,w)
                expected=commit_ring(cpu_ring,cpu_tags,cpu_lengths,staged,cpu_advance)
                graph.replay();sync()
                actual=(ring.cpu(),tags.cpu(),lengths.cpu())
                passed=all(torch.equal(a,e)for a,e in zip(actual,expected))and torch.equal(mask.cpu(),expected_mask)
                record=dict(component='ring',start=start,advance=advances,replay=replay,passed=passed)
                records.append(record);assert passed,record
                cpu_ring,cpu_tags,cpu_lengths=expected
    del graph;sync()
    base=torch.zeros((3,4,3),dtype=torch.float32,device='hpu')
    anchors=torch.tensor([2,4,1],dtype=torch.int32,device='hpu')
    w1=torch.eye(5,device='hpu');w2cpu=torch.zeros((3,5));w2cpu[1,2]=10;w2cpu[2,4]=10;w2cpu[0,1]=10
    w2=w2cpu.to('hpu');mapping=torch.tensor([2,4,1],dtype=torch.int32,device='hpu')
    sync();graph=torch.hpu.HPUGraph()
    with torch.hpu.graph(graph):drafts=dspark_greedy(base+0,anchors+0,w1,w2,mapping)
    sync()
    for changed in (False,True,False):
        cpu_base=torch.zeros((3,4,3))
        if changed:cpu_base[0,0,2]=20
        base.copy_(cpu_base);sync();graph.replay();sync()
        expected=dspark_greedy(cpu_base,torch.tensor([2,4,1]),torch.eye(5),w2cpu,torch.tensor([2,4,1]))
        actual=drafts.cpu();passed=torch.equal(actual,expected)
        records.append(dict(component='dspark_markov',changed=changed,ids=actual.tolist(),passed=passed));assert passed
    del graph;sync()
(out/'result.json').write_text(json.dumps(dict(status='PASS_SYNTHETIC_DEVICE_STATE',records=records,
    scope='SWA transaction and dense greedy Markov component only; no drafter checkpoint, complete native cycle, model quality or TPS claim'),indent=2)+'\n')
