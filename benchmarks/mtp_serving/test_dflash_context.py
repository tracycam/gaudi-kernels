import unittest
import torch
from dflash_context import CommittedContext


class Context(unittest.TestCase):
    def test_ragged_rejection_wrap_and_request_reset(self):
        ring=CommittedContext(layers=2,batch=3,window=8,kv_heads=1,dim=2,device='cpu')
        reference=[{}for _ in range(3)]
        for start,counts in ((0,[4,2,0]),(4,[4,3,4]),(8,[4,0,1]),(12,[1,4,2])):
            pos=torch.arange(start,start+4).expand(3,-1).int();count=torch.tensor(counts)
            pair=tuple(tuple((pos[:,:,None,None]+100*layer+10*kind).expand(3,4,1,2).bfloat16()for kind in range(2))for layer in range(2))
            ring.commit(pair,pos,count)
            for b,n in enumerate(counts):
                for j in range(n):reference[b][int(pos[b,j])%8]=int(pos[b,j])
            kv,positions,valid=ring.state()
            for b,want in enumerate(reference):
                self.assertEqual({i:int(positions[b,i])for i in range(8)if valid[b,i]},want)
                for slot,value in want.items():
                    for layer in range(2):
                        for kind in range(2):self.assertTrue(torch.equal(kv[layer][kind][b,slot],torch.full((1,2),value+100*layer+10*kind,dtype=torch.bfloat16)))
        before=tuple(tuple(x.clone()for x in pair)for pair in ring.state()[0]);old=ring.state()[1].clone()
        ring.commit(pair,pos,torch.zeros(3,dtype=torch.int32))
        self.assertTrue(torch.equal(old,ring.state()[1]))
        self.assertTrue(all(torch.equal(x,y)for p,q in zip(before,ring.state()[0])for x,y in zip(p,q)))
        ring.reset(torch.tensor([False,True,False]))
        self.assertFalse(ring.state()[2][1].any())
        self.assertTrue(torch.equal(old[[0,2]],ring.state()[1][[0,2]]))


if __name__=='__main__':unittest.main()
