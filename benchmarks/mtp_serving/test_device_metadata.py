import unittest
import torch
from device_metadata import query_metadata


class DeviceMetadata(unittest.TestCase):
    def test_causal_pages_slots_and_window_for_ragged_rows(self):
        pages=torch.arange(3*40,dtype=torch.int32).reshape(3,40)+10
        for start in (0,1,125,126,127,128,129,4094,4095):
            for counts in ((1,2,4),(4,4,4),(0,3,1)):
                lens=torch.tensor([start,start+2,start+4],dtype=torch.int32)
                result=query_metadata(pages,lens,torch.tensor(counts),4,128,128,999,1000)
                self.assertFalse(result['fault'].any())
                for b in range(3):
                    for j in range(4):
                        q=start+2*b+j;row=b*4+j
                        slot=result['slot_mapping'][row].item()
                        for family in ('','window_'):
                            selected=result[family+'block_groups']==row
                            actual=result[family+'block_list'][selected].tolist()
                            first=max(0,(q-127)//128)if family else 0
                            expected=pages[b,first:q//128+1].tolist()if j<counts[b]else []
                            self.assertEqual(actual,expected)
                            usage=result[family+'block_usage'][selected].int().tolist()
                            self.assertEqual(usage,[128]*(len(expected)-1)+[q%128+1]if expected else [])
                        if j<counts[b]:self.assertEqual(slot,pages[b,q//128].item()*128+q%128)
                        else:self.assertEqual(slot//128,999)

    def test_fault_does_not_address_live_storage(self):
        for lengths,counts in (([255,0],[4,1]),([-1,0],[1,0]),([0,0],[5,1])):
            result=query_metadata(torch.tensor([[1,2],[3,4]]),torch.tensor(lengths),torch.tensor(counts),4,128,128,99,100)
            self.assertTrue(result['fault'][0]);self.assertFalse(result['fault'][1])
            self.assertTrue((result['slot_mapping'][:4]//128==99).all())
            self.assertFalse(((result['block_groups']>=0)&(result['block_groups']<4)).any())
        for bad in (-1,99,100):
            result=query_metadata(torch.tensor([[1,bad]]),torch.tensor([127]),torch.tensor([4]),4,128,128,99,100)
            self.assertTrue(result['fault'][0])


if __name__=='__main__':unittest.main()
