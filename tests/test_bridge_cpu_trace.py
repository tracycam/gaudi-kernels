import unittest
from tools.validation.executor.analyze_bridge_cpu_trace import analyze


def event(name, phase, time, thread, **extra):
    return dict(name=name, ph=phase, ts=time, pid=1, tid=thread, **extra)


class BridgeCPUTraceTest(unittest.TestCase):
    def test_nested_launch_and_different_submission_thread(self):
        trace = dict(traceEvents=[
            event('HPUGraph.replay','X',0,10,dur=20),
            event('SubmitCollectivePlan','B',1,10),
            event('NativePlanSubmit','B',4,20),
            event('Launch','B',5,20),
            event('launch','B',6,20),
            event('LaunchSynapseSubmit','X',7,20,dur=1),
            event('LaunchAddressLease','X',6,30,dur=1),  # Other thread.
            event('launch','E',9,20),
            event('Launch','E',10,20),
            event('NativePlanSubmit','E',14,20),
            event('SubmitCollectivePlan','E',18,10)])
        result = analyze(trace, command_details=True)
        self.assertEqual(result['unmatched'], [])
        row = result['queued_submissions'][0]
        self.assertEqual((row['caller_tid'],row['submission_tid']),(10,20))
        self.assertEqual(row['queue_entry_to_submit_us'],3)
        self.assertEqual(row['submit_us'],10)
        self.assertEqual(row['submit_end_to_caller_return_us'],4)
        # Parent Launch includes child launch, counted once in the submit total.
        self.assertEqual(row['launcher_inclusive_us'],5)
        self.assertEqual(row['commands'][0]['compute_launch_us'],3)
        self.assertEqual(row['launch_stages']['LaunchSynapseSubmit'],
                         dict(count=1, inclusive_us=1))
        self.assertEqual(row['launch_stages']['LaunchAddressLease']['count'],0)
        self.assertEqual(result['stages']['HPUGraph.replay']['inclusive_us']['median'],20)

    def test_compute_only_is_not_mislabeled_as_queue_delay(self):
        trace = dict(traceEvents=[event('NativePlanSubmit','B',0,10),
                                 event('NativePlanSubmit','E',4,10)])
        result = analyze(trace)
        self.assertEqual(result['queued_submissions'], [])
        self.assertEqual(result['stages']['NativePlanSubmit']['inclusive_us']['median'],4)

    def test_command_details_exclude_other_threads_and_noncontained_spans(self):
        trace = dict(traceEvents=[
            event('SubmitCollectivePlan','X',0,10,dur=30),
            event('NativePlanSubmit','X',2,20,dur=25),
            event('Launch','X',5,20,dur=10),
            event('Launch','X',20,20,dur=5),
            event('launch','X',6,21,dur=3),  # Another thread.
            event('launch','X',4,20,dur=3),  # Began outside command.
            event('launch','X',7,20,dur=4),
            event('launch','X',14,20,dur=3),  # Ends outside command.
            event('launch','X',21,20,dur=2)])
        commands = analyze(trace, command_details=True)['queued_submissions'][0]['commands']
        self.assertEqual([c['compute_launch_count'] for c in commands], [1, 1])
        self.assertEqual([c['compute_launch_us'] for c in commands], [4, 2])

    def test_partial_or_mismatched_spans_are_reported(self):
        trace = dict(traceEvents=[event('end','E',0,10),event('a','B',1,10),
                                 event('wrong','E',2,10),event('open','B',3,20)])
        self.assertEqual(len(analyze(trace)['unmatched']),3)


if __name__ == '__main__':
    unittest.main()
