# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import unittest

import torch

from gaudi_kernels.engine.token_batch import QueryBatch, RequestQueries
from gaudi_kernels.serving.draft.cycle import DFlashCycle
from gaudi_kernels.serving.executor.packed_attention import PackedKVSession
from gaudi_kernels.serving.executor.packed_target import PackedTargetResult


class DFlashCycleTests(unittest.TestCase):
    def make_cycle(self, request_ids=('a',)):
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
                # The shared mock target head predicts hidden+1. DFlash must
                # reconstruct the masked token at the same position; including
                # its anchor output would therefore cause immediate rejection.
                return (noise-1).to(torch.bfloat16)

            def project_context(self, features, positions):
                values = features[..., :4].unsqueeze(2)
                return ((values, values.clone()), (values.clone(), values.clone()))

        return DFlashCycle(Target(), Draft(), request_ids)

    def test_device_token_edge_complete_acceptance_budget_and_context_commit(self):
        cycle = self.make_cycle()
        target = cycle.target
        anchor = torch.tensor([10], dtype=torch.int32)
        for start, budget, want in ((0, 9, (11, 12, 13, 14)), (4, 2, (15, 16))):
            schedule = QueryBatch((RequestQueries('a', 4, start, 'verify'),))
            meta = target.session.prepare(schedule)
            result = cycle.run_prepared(schedule, meta, anchor, torch.tensor([budget], dtype=torch.int32))
            delivered = cycle.finish_at_output_boundary(result)
            self.assertEqual(delivered.requests[0].emitted_tokens, want)
            self.assertEqual(delivered.requests[0].committed_queries, len(want))
            self.assertEqual(len(target.session.committed['a']), start+len(want))
            anchor = result.next_anchor_ids
            with self.assertRaises(ValueError):
                cycle.finish_at_output_boundary(result)
        _, positions, valid = cycle.context.state()
        self.assertEqual(positions[valid].tolist(), list(range(6)))
        self.assertEqual(anchor.tolist(), [16])

    def test_ragged_denoised_rows_and_zero_draft_bypass(self):
        cycle = self.make_cycle(('a', 'b'))
        schedule = QueryBatch((RequestQueries('a', 1, 0, 'decode'), RequestQueries('b', 3, 0, 'verify')))
        result = cycle.run_prepared(schedule, cycle.target.session.prepare(schedule),
            torch.tensor([10, 30], dtype=torch.int32), torch.tensor([10, 10], dtype=torch.int32))
        self.assertEqual(result.proposed_ids.tolist(), [[-1, -1], [31, 32]])
        output = cycle.finish_at_output_boundary(result)
        self.assertEqual([r.emitted_tokens for r in output.requests], [(11,), (31, 32, 33)])
        self.assertEqual(result.next_anchor_ids.tolist(), [11, 33])
        _, positions, valid = cycle.context.state()
        self.assertEqual([positions[i][valid[i]].tolist() for i in range(2)], [[0], [0, 1, 2]])

        # Pure decode has no masked row. It must not invoke the drafter or
        # accidentally put an anchor reconstruction through the proposal head.
        def forbidden(*args):
            raise AssertionError('Zero-draft cycle invoked the drafter')
        cycle.draft.noise = forbidden
        decode = QueryBatch((RequestQueries('a', 1, 1, 'decode'), RequestQueries('b', 1, 3, 'decode')))
        next_result = cycle.run_prepared(decode, cycle.target.session.prepare(decode),
            result.next_anchor_ids, torch.tensor([10, 10], dtype=torch.int32))
        self.assertEqual(next_result.proposed_ids.shape, (2, 0))
        delivered = cycle.finish_at_output_boundary(next_result)
        self.assertEqual([r.emitted_tokens for r in delivered.requests], [(12,), (34,)])
