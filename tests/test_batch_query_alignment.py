from types import SimpleNamespace
import unittest
from tools.validation.executor.batch_quality import attach_owners,index_rows,generate_with_ids


class QueryAlignmentTests(unittest.TestCase):
    def test_internal_ids_are_mapped_at_admission_and_hook_is_restored(self):
        def assign(req):
            req.external_req_id=req.request_id
            req.request_id+='-random-suffix'
        processor=SimpleNamespace(assign_request_id=assign)
        def generate(*args,**kwargs):
            req=SimpleNamespace(request_id='client-id')
            processor.assign_request_id(req)
            return [SimpleNamespace(request_id=req.external_req_id)]
        llm=SimpleNamespace(llm_engine=SimpleNamespace(input_processor=processor),generate=generate)
        outputs,mapping=generate_with_ids(llm,[],None)
        frame={'request_ids':['client-id-random-suffix']}
        self.assertEqual(attach_owners([frame],outputs,mapping)[0]['request_indices'],[0])
        self.assertIs(processor.assign_request_id,assign)

    def test_staggered_and_grouped_batches_keep_every_request_even_with_equal_token_ids(self):
        first=[{'path':'a','request_ids':['r0'],'request_keys':['prompt0'],
                'logits_indices':[0],'input_ids':[77],'positions':[125]},
               {'path':'b','request_ids':['r0','r1',None],'request_keys':['prompt0','prompt1',None],
                'logits_indices':[0,1,0],'input_ids':[77,77,0],'positions':[126,125,0]}]
        second=[{'path':'c','request_ids':['s0','s1'],'request_keys':['prompt0','prompt1'],
                 'logits_indices':[0,1],'input_ids':[77,77],'positions':[125,125]},
                {'path':'d','request_ids':['s0'],'request_keys':['prompt0'],
                 'logits_indices':[0],'input_ids':[77],'positions':[126]}]
        a=index_rows(attach_owners(first,[SimpleNamespace(request_id='r0'),SimpleNamespace(request_id='r1')]))
        b=index_rows(attach_owners(second,[SimpleNamespace(request_id='s0'),SimpleNamespace(request_id='s1')]))
        self.assertEqual(set(a),set(b))
        self.assertEqual(len(a),3)
        self.assertEqual(a[(1,'prompt1',77,125)][1],1)

    def test_duplicate_query_is_not_silently_overwritten(self):
        frame={'path':'f','request_indices':[0],'request_keys':['p'],'logits_indices':[0],
               'input_ids':[77],'positions':[125]}
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            index_rows([frame,frame])
