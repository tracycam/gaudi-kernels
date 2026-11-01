# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from gaudi_kernels.serving.executor.greedy_verify import verify_ids


class PackedGreedyTests(unittest.TestCase):
    def test_every_prefix_eos_budget_and_ragged_request(self):
        for k in (0, 1, 3, 7):
            for prefix in range(k+1):
                for budget in (0, 1, 2, k+1, k+3):
                    drafts = list(range(10, 10+k))
                    predictions = drafts+[77]
                    if prefix < k:
                        predictions[prefix] = 88
                    expected = drafts[:prefix]+[predictions[prefix]]
                    for eos in ((), (10,), (77, 88)):
                        want = expected[:budget]
                        for i, token in enumerate(want):
                            if token in eos:
                                want = want[:i+1]
                                break
                        # A second independent no-draft request shares the call.
                        result = verify_ids(torch.tensor([99]+drafts+[42], dtype=torch.int32),
                            torch.tensor(predictions+[66], dtype=torch.int32), (k+1, 1),
                            torch.tensor([budget, 1], dtype=torch.int32), eos_ids=eos)
                        self.assertEqual(result.emitted_ids[0].tolist(), want+[-1]*(k+1-len(want)))
                        self.assertEqual(result.committed_input_counts.tolist(), [len(want), 1])
                        self.assertEqual(result.emitted_counts.tolist(), [len(want), 1])
                        self.assertEqual(result.matched_draft_counts.tolist(), [prefix, 0])
                        self.assertEqual(result.emitted_ids[1].tolist(), [66]+[-1]*k)
                        self.assertTrue(result.valid.all())

    def test_invalid_budget_cannot_promote_kv(self):
        result = verify_ids(torch.tensor([11, 12]), torch.tensor([12, 13]), (2,), torch.tensor([-1]))
        self.assertEqual(result.committed_input_counts.tolist(), [0])
        self.assertEqual(result.emitted_ids.tolist(), [[-1, -1]])
        self.assertEqual(result.valid.tolist(), [False])
