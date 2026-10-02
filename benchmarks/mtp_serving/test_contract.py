import unittest
import torch
from metrics import acceptance,budget,routes
from protocol import greedy_accept,KVTransaction

class Contract(unittest.TestCase):
    def test_accept_reject_eos_and_budget(self):
        for k in (1,3,7):
            for prefix in range(k+1):
                for remaining in (0,1,k,k+1):
                    for eos_position in range(-1,k+1):
                        drafts=torch.arange(10,10+k,dtype=torch.int32)[None,:]
                        target=torch.cat((drafts.clone(),torch.tensor([[99]],dtype=torch.int32)),1)
                        if prefix<k:target[0,prefix]=77
                        expected=drafts[0,:prefix].tolist()+[target[0,prefix].item()]
                        eos=expected[eos_position]if eos_position<len(expected)and eos_position>=0 else 12345
                        expected=expected[:remaining]
                        if eos in expected:expected=expected[:expected.index(eos)+1]
                        out,n=greedy_accept(drafts,target,torch.tensor([k],dtype=torch.int32),torch.tensor([remaining],dtype=torch.int32),eos)
                        self.assertEqual(n.tolist(),[len(expected)]);self.assertEqual(out[0].tolist(),expected+[-1]*(k+1-len(expected)))
    def test_lengths_and_sequences(self):
        d=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.int32);t=torch.tensor([[1,2,3,9],[4,5,6,8]],dtype=torch.int32)
        out,n=greedy_accept(d,t,torch.tensor([0,2],dtype=torch.int32),torch.tensor([8,8],dtype=torch.int32),99)
        self.assertEqual(out.tolist(),[[1,-1,-1,-1],[4,5,6,-1]]);self.assertEqual(n.tolist(),[1,3])
    def test_kv_boundaries_reject_and_abort(self):
        for length in (0,1,126,127,128,129,255,256,4095,4096):
            for k in (2,4,8):
                for keep in range(k+1):
                    prefix=list(range(length));values=list(range(10000,10000+k));cache=KVTransaction(128,prefix);before=dict(cache.committed)
                    cache.begin(values)
                    for q in range(k):self.assertEqual(cache.view(q),(prefix+values[:q+1])[-128:])
                    self.assertEqual(cache.committed,before)
                    cache.abort();self.assertEqual(cache.committed,before)
                    cache.begin(values);cache.commit(keep);truth=prefix+values[:keep]
                    self.assertEqual(list(cache.committed.values()),truth[-128:]);self.assertEqual(cache.length,len(truth))
                    cache.begin([20000]);self.assertEqual(cache.view(0),(truth+[20000])[-128:])
    def test_tau_not_acceptance_ratio(self):
        data=[dict(requests=[dict(drafts=[1,2,3],output_ids=ids,accepted_drafts=len(ids)-1,emitted=len(ids))])for ids in([8],[1,9],[1,2,3,7])]
        value=acceptance(data);self.assertAlmostEqual(value['tau'],7/3)
        self.assertEqual([v['alpha']for v in value['conditional_alpha']],[2/3,1/2,1.])
        self.assertAlmostEqual(budget(2.3,25)['per_request_tps'],92)
    def test_route_padding_and_coverage(self):
        rows=[list(range(8)),list(range(8)),list(range(8,16))]
        f=dict(requests=[('a',2),('b',1)],layers={'1':rows,'2':rows})
        r=routes(f,[1,2]);self.assertEqual(r[0]['m_histogram'],{2:8,1:8});self.assertEqual(r[0]['per_request'][0]['U'],8)
        with self.assertRaises(ValueError):routes(f,[1,2,3])
        f['layers']['1']=rows+[list(range(8))]
        with self.assertRaises(ValueError):routes(f,[1,2])

if __name__=='__main__':unittest.main()
