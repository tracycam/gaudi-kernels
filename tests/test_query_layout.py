import unittest
from gaudi_kernels.serving.executor.query_layout import DecodeQueries


class QueryLayoutTests(unittest.TestCase):
    def test_ragged_verify_at_page_boundary_has_every_query_and_no_padding_owner(self):
        layout=DecodeQueries(('r0','r1','r2'),(4,2,1),(126,127,400),16,True)
        positions=[126,127,128,129,127,128,0,0,400,0,0,0,0,0,0,0]
        selected=[0,1,2,3,4,5,-1,-1,8,-1,-1,-1,12,-1,-1,-1]
        owners=layout.validate(positions,[0,1,2,3,4,5,8],selected)
        self.assertEqual(owners,['r0']*4+['r1']*2+[None]*2+['r2']+[None]*7)
        with self.assertRaisesRegex(ValueError,'visibility'):
            layout.validate(positions,[0,1,2,3,4,5,6,8],selected)
        selected[4]=0
        with self.assertRaisesRegex(ValueError,'another request/query'):
            layout.validate(positions,[0,1,2,3,4,5,8],selected)

    def test_decode_padding_zero_is_not_an_additional_request(self):
        layout=DecodeQueries(('a','b','c'),(1,1,1),(125,126,127),4,False)
        self.assertEqual(layout.validate([125,126,127,0],[0,1,2],[0,1,2,0]),['a','b','c',None])
        with self.assertRaisesRegex(ValueError,'cursor'):
            layout.validate([125,125,127,0],[0,1,2],[0,1,2,0])

    def test_query_count_does_not_stand_in_for_request_count(self):
        layout=DecodeQueries(('single',),(4,),(128,),4,True)
        self.assertEqual(len(layout.active_rows),4)
        self.assertEqual(len(layout.request_ids),1)
        with self.assertRaises(ValueError):
            DecodeQueries(('duplicate','duplicate'),(1,1),(0,0),2,False)
