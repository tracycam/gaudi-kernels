import unittest

import torch

from gaudi_kernels.serving.diagnostics.quality import compare_logits
from gaudi_kernels.serving.executor.production_quality import compare_logits as historical_entry


class QualityApiTests(unittest.TestCase):
    def test_historical_tools_call_the_same_comparator(self):
        self.assertIs(historical_entry, compare_logits)
        logits = torch.tensor([[0., 1., -1.]])
        check = historical_entry(logits, logits, contract='fp32_arithmetic_v1')
        self.assertTrue(check['pass'])
        self.assertEqual(check['max_kl_ref_to_candidate'], 0)
        self.assertEqual(check['top1_match_fraction'], 1)

    def test_divergent_and_nonfinite_outputs_fail(self):
        logits = torch.tensor([[20., -20.]])
        self.assertFalse(compare_logits(logits, -logits, contract='fp32_arithmetic_v1')['pass'])
        self.assertFalse(compare_logits(logits, torch.tensor([[float('nan'), 0.]]))['pass'])
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            compare_logits(logits, logits[:, :1])

    def test_fp32_contract_cannot_bypass_l2_with_a_softmax_invariant_shift(self):
        logits=torch.tensor([[0.,1.,-1.]])
        for contract in ('legacy','fp32_arithmetic_v1'):
            check=compare_logits(logits,logits+.1,contract=contract)
            self.assertLess(check['max_kl_ref_to_candidate'],.01)
            self.assertGreater(check['relative_l2'],.02)
            self.assertFalse(check['pass'])
            self.assertEqual(check['gate']['relative_l2'],.02)

    def test_good_positions_cannot_hide_one_bad_row(self):
        logits=torch.tensor([[0.,1.,-1.]]).repeat(100,1)
        candidate=logits.clone();candidate[0]+=.1
        check=compare_logits(logits,candidate,contract='fp32_arithmetic_v1')
        self.assertLess(check['relative_l2'],.02)
        self.assertGreater(check['max_row_relative_l2'],.02)
        self.assertFalse(check['pass'])


if __name__ == '__main__':
    unittest.main()
