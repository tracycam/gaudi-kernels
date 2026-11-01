import json
from pathlib import Path
import tempfile
import unittest

from tools.validation.executor.analyze_bridge_device_trace import analyze, engine_family, from_sdk_jsonl, union


class DeviceTraceTest(unittest.TestCase):
    def test_scheduler_tracks_are_not_physical_engines(self):
        for name in ('ENG ARC [DCORE0_TPC0]', 'ENG ARC [DCORE0_MME]',
                     'ENG ARC [PDMA0]', 'DMA Kernel Driver'):
            self.assertIsNone(engine_family(name))
        self.assertEqual(engine_family('[D0] TPC 0'), 'TPC')
        self.assertEqual(engine_family('[D2] MME'), 'MME')
        self.assertEqual(engine_family('[D1] Internal DMA 1 (HCL dedicated)'), 'EDMA')
        self.assertEqual(engine_family('DMA 0'), 'PDMA')
        self.assertEqual(engine_family('Port 9'), 'NIC')

    def test_scheduler_activity_does_not_fill_compute_idle_time(self):
        document = {'traceEvents': [
            dict(ph='M',pid=1,tid=1,name='thread_name',args={'name':'[D0] TPC 0'}),
            dict(ph='M',pid=1,tid=2,name='thread_name',args={'name':'ENG ARC [DCORE0_TPC0]'}),
            dict(ph='X',pid=1,tid=0,name='runner.sample_tokens',ts=0,dur=10),
            dict(ph='X',pid=1,tid=1,name='a',cat='kernel',ts=1,dur=2,args={'recipeId':1}),
            dict(ph='X',pid=1,tid=2,name='scheduler',cat='kernel',ts=1,dur=8,args={'recipeId':1}),
            dict(ph='X',pid=1,tid=1,name='b',cat='kernel',ts=7,dur=2,args={'recipeId':1})]}
        row = analyze(document, skip_first=0)['rows'][0]
        self.assertEqual(row['compute_span_us'], 8)
        self.assertEqual(row['compute_union_us'], 4)
        self.assertEqual(row['observed_engine_idle_within_compute_span_us'], 4)

    def test_overlap_is_counted_once(self):
        self.assertEqual(union([(2, 7), (1, 4), (7, 8), (10, 11)]), [[1, 8], [10, 11]])

    def test_engine_pairing_is_not_a_nested_cpu_stack(self):
        # MME operations may overlap on one engine trace. Pair by event context
        # rather than pairing the first end with the most recent begin.
        events = [dict(ph='M', ts=0, pid=1, tid=2, name='thread_name', args={'name':'[D0] MME'})]
        for phase, ts, name, context in [('B',1,'a',1),('B',2,'b',2),('E',4,'a',1),('E',8,'b',2)]:
            events.append(dict(ph=phase,ts=ts,pid=1,tid=2,id=context,name=name,
                               cat='SP 0',args={'recipe_id': 10}))
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'events.jsonl'
            p.write_text(''.join(json.dumps(e)+'\n' for e in events))
            result=from_sdk_jsonl(p)
        self.assertEqual(result['unmatched_sdk_events'], [])
        self.assertEqual([(e['name'],e['dur']) for e in result['traceEvents'] if e['ph']=='X'],
                         [('a',3),('b',6)])


if __name__ == '__main__':
    unittest.main()
