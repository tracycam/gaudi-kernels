import unittest
import torch
from device_state import commit_ring, dspark_greedy, visibility


class DeviceState(unittest.TestCase):
    def test_partial_commit_keeps_live_window_and_rejects_staging(self):
        for start in (0,1,126,127,128,129,255,256,4095,4096):
            b,w,t = 3,128,8
            ring=torch.full((b,w,2),-999,dtype=torch.int32)
            tags=torch.full((b,w),-1,dtype=torch.int64)
            lengths=torch.tensor([start,start+2,start+4])
            for row,n in enumerate(lengths.tolist()):
                for p in range(max(0,n-w),n):
                    tags[row,p%w]=p;ring[row,p%w]=torch.tensor([p,row])
            original=ring.clone()
            staging=torch.stack([torch.stack([torch.tensor([n+j,r],dtype=ring.dtype)for j in range(t)])for r,n in enumerate(lengths.tolist())])
            mask=visibility(tags,lengths,t,w)
            keys=torch.cat((tags,lengths[:,None]+torch.arange(t)),1)
            for row,n in enumerate(lengths.tolist()):
                for q in range(t):
                    self.assertEqual(sorted(keys[row][mask[row,q]].tolist()),list(range(max(0,n+q-w+1),n+q+1)))
            for accepted in ((0,1,8),(8,3,0),(1,1,1)):
                advance=torch.tensor(accepted);new,tags2,n2=commit_ring(ring,tags,lengths,staging,advance)
                self.assertTrue(torch.equal(ring,original),'input ring mutated before graph commit')
                for row,end in enumerate(n2.tolist()):
                    for p in range(max(0,end-w),end):
                        self.assertEqual(tags2[row,p%w].item(),p)
                        self.assertEqual(new[row,p%w].tolist(),[p,row])
                    self.assertFalse(((tags2[row]>=end)&(tags2[row]>=0)).any())

    def test_markov_head_uses_previous_remapped_token(self):
        # Draft IDs 0,1,2 map to target IDs 2,4,1. Identity transition
        # coordinates make the head choose a 2->4->1->2 cycle.
        w1=torch.eye(5);w2=torch.zeros((3,5));w2[1,2]=10;w2[2,4]=10;w2[0,1]=10
        base=torch.zeros((2,4,3));mapping=torch.tensor([2,4,1]);anchor=torch.tensor([2,4])
        got=dspark_greedy(base,anchor,w1,w2,mapping)
        self.assertEqual(got.tolist(),[[4,1,2,4],[1,2,4,1]])
        # Deliberately changing an earlier choice must change later logits.
        base[0,0,2]=20
        self.assertEqual(dspark_greedy(base,anchor,w1,w2,mapping)[0].tolist(),[1,2,4,1])


if __name__=='__main__':unittest.main()
