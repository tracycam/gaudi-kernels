import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch; also run in the qualified remote environment')
class BatchCommitTests(unittest.TestCase):
    def test_every_prefix_eos_budget_and_inactive_row_against_scalar_reference(self):
        import torch
        from gaudi_kernels.serving.executor.batch_state import verify_greedy
        cases = 0
        for k in (0,1,3,7):
            for prefix in range(k+1):
                for budget in range(k+3):
                    for eos_at in range(-1,k+1):
                        drafts = list(range(10,10+k))
                        target = drafts+[77]
                        if prefix < k:
                            target[prefix] = 88
                        if eos_at >= 0:
                            # Change both arrays so EOS may be an accepted draft.
                            target[eos_at] = 99
                            if eos_at < k:
                                drafts[eos_at] = 99
                        expected = []
                        for i, value in enumerate(drafts):
                            if value != target[i]:
                                break
                            expected.append(value)
                        expected.append(target[len(expected)])
                        expected = expected[:budget]
                        if 99 in expected:
                            expected = expected[:expected.index(99)+1]
                        out = verify_greedy(torch.tensor([target]*2,dtype=torch.int32),
                            torch.tensor([drafts]*2,dtype=torch.int32), torch.tensor([126,500]),
                            torch.tensor([k,k]), torch.tensor([True,False]),
                            remaining=torch.tensor([budget,budget]), eos_id=99)
                        self.assertEqual(out.tokens[0,:out.token_count[0]].tolist(), expected)
                        self.assertEqual(out.token_count.tolist(), [len(expected),0])
                        self.assertEqual(out.committed_kv_length.tolist(), [126+len(expected),500])
                        self.assertEqual(out.next_token.tolist(), [expected[-1] if expected else 0,0])
                        cases += 1
        self.assertGreater(cases, 700)
