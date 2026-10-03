# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import unittest

import torch

from gaudi_kernels.engine.token_batch import QueryBatch, RequestQueries
from gaudi_kernels.serving.draft.cycle import DFlashCycle
from gaudi_kernels.serving.executor.packed_attention import PackedKVSession
from gaudi_kernels.serving.executor.packed_target import PackedTargetResult


class DFlashCycleTests(unittest.TestCase):
    def test_device_token_edge_complete_acceptance_budget_and_context_commit(self):
        # CPU substitutes validate wiring/ownership, not checkpoint accuracy.
        class Model:
            def embed_input_ids(self, ids):
                return ids[:, None].expand(-1, 4).to(torch.bfloat16)

            def compute_logits(self, hidden):
                logits = torch.zeros(hidden.shape[0], 100)
                logits.scatter_(1, (hidden[:, :1].int()+1).long(), 1)
                return logits

        class Target:
            feature_layers = (0, 1)
            def __init__(self):
                self.runner = SimpleNamespace(device='cpu')
                self.model = Model()
                self.session = PackedKVSession({'layer': (1, 4, 4)}, 32, device='cpu')

            def execute_prepared(self, schedule, metadata, binding):
                binding.validate(schedule)
                hidden = self.model.embed_input_ids(binding.ids)
                return PackedTargetResult(hidden, self.model.compute_logits(hidden), schedule.logits_owners,
                                          metadata, (hidden.clone(), hidden.clone()), self.feature_layers)

            def commit(self, result, counts):
                self.session.commit(result.transaction, counts)

        class Draft:
            spec = SimpleNamespace(target_layers=(0, 1), hidden=4, layers=2, block=4, window=8, kv_heads=1, dim=4)
            def noise(self, anchor):
                return anchor[:, None] + torch.arange(4)[None, :, None]

            def __call__(self, noise, *args):
                return noise.to(torch.bfloat16)

            def project_context(self, features, positions):
                values = features[..., :4].unsqueeze(2)
                return ((values, values.clone()), (values.clone(), values.clone()))

        target = Target()
        cycle = DFlashCycle(target, Draft(), ('a',))
        anchor = torch.tensor([10], dtype=torch.int32)
        for start, budget, want in ((0, 9, (11, 12, 13, 14)), (4, 2, (15, 16))):
            schedule = QueryBatch((RequestQueries('a', 4, start, 'verify'),))
            meta = target.session.prepare(schedule)
            result = cycle.run_prepared(schedule, meta, anchor, torch.tensor([budget], dtype=torch.int32))
            delivered = cycle.commit_after_delivery(result)
            self.assertEqual(delivered.requests[0].emitted_tokens, want)
            self.assertEqual(delivered.requests[0].committed_queries, len(want))
            self.assertEqual(len(target.session.committed['a']), start+len(want))
            anchor = result.next_anchor_ids
            with self.assertRaises(ValueError):
                cycle.commit_after_delivery(result)
        _, positions, valid = cycle.context.state()
        self.assertEqual(positions[valid].tolist(), list(range(6)))
        self.assertEqual(anchor.tolist(), [16])
