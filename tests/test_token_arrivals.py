from types import SimpleNamespace
import unittest
from tools.validation.executor.token_arrivals import run, summarize


class TokenArrivalTests(unittest.TestCase):
    def test_speculative_groups_are_counted_at_real_arrival_time(self):
        events=[dict(request_id='a',arrival_ns=10,new_tokens=4,total_tokens=4,finished=False),
                dict(request_id='b',arrival_ns=20,new_tokens=4,total_tokens=4,finished=False),
                dict(request_id='a',arrival_ns=30,new_tokens=3,total_tokens=7,finished=True),
                dict(request_id='b',arrival_ns=40,new_tokens=2,total_tokens=6,finished=True)]
        doc=summarize(events,['a','b'])
        self.assertEqual(doc['per_request'][0]['measured_tokens'],3)
        self.assertEqual(doc['per_request'][0]['decode_span_ns'],20)
        self.assertEqual(doc['aggregate']['tokens'],5)
        self.assertEqual(doc['aggregate']['begin_ns'],20)
        self.assertEqual(doc['aggregate']['end_ns'],40)
        self.assertEqual(doc['aggregate']['per_request'][0]['tokens'],3)
        self.assertEqual(doc['aggregate']['per_request'][0]['span_ns'],10)

    def test_final_only_output_cannot_be_reported_as_decode_tps(self):
        doc=summarize([dict(request_id='a',arrival_ns=100,new_tokens=8,total_tokens=8,finished=True)],['a'])
        self.assertIsNone(doc['per_request'][0]['tps'])
        self.assertIsNone(doc['aggregate']['tps'])

    def test_every_request_and_completion_is_required(self):
        with self.assertRaisesRegex(ValueError,'completed request'):
            summarize([dict(request_id='a',arrival_ns=10,new_tokens=1,total_tokens=1,finished=False)],['a','b'])

    def test_engine_ids_and_full_output_are_preserved(self):
        def output(rid,values,finished):
            return SimpleNamespace(request_id=rid,outputs=[SimpleNamespace(token_ids=values)],finished=finished)
        queue=[[output('case-0',[7],False),output('case-1',[8],False)],
               [output('case-0',[7,9],True),output('case-1',[8,10],True)]]
        added=[]
        engine=SimpleNamespace(has_unfinished_requests=lambda:bool(added and queue),
            add_request=lambda rid,prompt,params:added.append(rid) or rid+'-internal',
            step=lambda:queue.pop(0),abort_request=lambda *args,**kwargs:None)
        clock=iter((100,110,120,130)).__next__
        outputs,doc=run(engine,[{},{}],SimpleNamespace(clone=lambda:None),tag='case',warmup_tokens=1,clock=clock)
        self.assertEqual([o.outputs[0].token_ids for o in outputs],[[7,9],[8,10]])
        self.assertEqual(doc['summary']['aggregate']['tokens'],2)
        self.assertEqual(doc['steps'],2)

    def test_non_cumulative_output_aborts_owned_requests(self):
        queue=[[SimpleNamespace(request_id='bad-0',outputs=[SimpleNamespace(token_ids=[7])],finished=False)],
               [SimpleNamespace(request_id='bad-0',outputs=[SimpleNamespace(token_ids=[9])],finished=True)]]
        added=[];aborted=[]
        engine=SimpleNamespace(has_unfinished_requests=lambda:bool(added and queue),
            add_request=lambda rid,prompt,params:added.append(rid) or 'owned-internal-id',
            step=lambda:queue.pop(0),abort_request=lambda ids,**kwargs:aborted.extend(ids))
        with self.assertRaisesRegex(ValueError,'cumulative'):
            run(engine,[{}],SimpleNamespace(clone=lambda:None),tag='bad')
        self.assertEqual(aborted,['owned-internal-id'])
