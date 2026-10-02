import unittest

from tools.validation.executor.moe_dispatch_gates import batch_capture_gate


class BatchCaptureGateTests(unittest.TestCase):
    def states(self, mode='broadcast', buckets=(1, 2, 4, 8)):
        return [{'serving_decode_batch_buckets': list(buckets), 'moe_dispatch': {
            'compact_rows': [1], 'python_capture_calls_by_rows': {'4': {mode: 69}}}}
            for _ in range(8)]

    def test_three_requests_require_the_declared_four_row_capture(self):
        gate = batch_capture_gate('bf16_fp32', 3, self.states())
        self.assertTrue(gate['pass'])
        self.assertEqual(gate['rows'], 4)
        self.assertFalse(batch_capture_gate('bf16_fp32', 3, self.states('compact'))['pass'])

    def test_unknown_or_inconsistent_bucket_coverage_cannot_pass(self):
        states = self.states()
        states[0].pop('serving_decode_batch_buckets')
        self.assertFalse(batch_capture_gate('bf16_fp32', 3, states)['pass'])
        self.assertFalse(batch_capture_gate('bf16_fp32', 3, self.states(buckets=(1, 2)))['pass'])


if __name__ == '__main__':
    unittest.main()
