import unittest
import torch
from gaudi_kernels.engine.token_batch import RequestTokens, TokenBatch
from gaudi_kernels.serving.executor.target_reference import fixture, packed_target, sequential_target


class PackedTargetTests(unittest.TestCase):
    def test_full_and_swa_mixed_forward_match_independent_token_reference(self):
        for window in (0, 128):
            args = fixture(sliding_window=window)
            packed = packed_target(*args)
            sequential = sequential_target(*args)
            torch.testing.assert_close(packed.hidden.float(), sequential.hidden.float(), rtol=.02, atol=.02)
            self.assertTrue(torch.isfinite(packed.hidden).all())
            self.assertEqual(packed.hidden.shape, (133, 32))
            self.assertEqual(packed.projection_calls, {'qkv': 1, 'out': 1})
            self.assertEqual(sequential.projection_calls, {'qkv': 133, 'out': 133})

    def test_future_queries_and_capacity_padding_cannot_change_prior_outputs(self):
        for window in (0, 128):
            batch, hidden, norm, qkv, out, histories, spec = fixture(sliding_window=window)
            before = packed_target(batch, hidden, norm, qkv, out, histories, spec)
            changed = hidden.clone()
            changed[3:5] = -changed[3:5].flip(-1)
            changed[133:] = float('nan')
            after = packed_target(batch, changed, norm, qkv, out, histories, spec)
            self.assertTrue(torch.equal(before.hidden[:3], after.hidden[:3]))
            self.assertTrue(torch.equal(before.hidden[5:], after.hidden[5:]))
            self.assertFalse(torch.equal(before.hidden[3:5], after.hidden[3:5]))

    def test_rejected_queries_are_absent_from_subsequent_decode_history(self):
        batch, hidden, norm, qkv, out, histories, spec = fixture(sliding_window=128)
        first = packed_target(batch, hidden, norm, qkv, out, histories, spec)
        for accepted_inputs in (0, 2, 4):
            past_k, past_v = histories[1]
            accepted_k = torch.cat((past_k, first.candidate_keys[1][:accepted_inputs]))
            accepted_v = torch.cat((past_v, first.candidate_values[1][:accepted_inputs]))
            next_batch = TokenBatch((RequestTokens('verify', (99,), 126 + accepted_inputs, 'decode'),))
            args = (next_batch, hidden[:1], norm, qkv, out, ((accepted_k, accepted_v),), spec)
            actual, reference = packed_target(*args), sequential_target(*args)
            self.assertTrue(torch.equal(actual.hidden, reference.hidden))
            if accepted_inputs < 4:
                contaminated = ((torch.cat((past_k, first.candidate_keys[1])),
                                  torch.cat((past_v, first.candidate_values[1]))),)
                with self.assertRaisesRegex(ValueError, 'History'):
                    packed_target(next_batch, hidden[:1], norm, qkv, out, contaminated, spec)
