# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import unittest

import torch

from gaudi_kernels.engine.token_batch import RequestTokens, TokenBatch
from gaudi_kernels.serving.executor.packed_attention import PackedKVSession, forward_packed, tiled_attention
from gaudi_kernels.serving.executor.packed_bindings import BoundKVSession, PackedInputBuffers


class PackedBindingTests(unittest.TestCase):
    def test_invalid_padding_cannot_leak_nan_values(self):
        q = torch.ones(1, 2, 4)
        k, v = torch.ones(4, 1, 4), torch.ones(4, 1, 3)
        k[1:], v[1:] = float('nan'), float('nan')
        result = tiled_attention(q, k, v, torch.tensor([0]), torch.arange(4), scale=.5,
                                 key_valid=torch.tensor([True, False, False, False]), tile_size=2)
        self.assertTrue(torch.equal(result, torch.ones(1, 2, 3)))

    def test_input_addresses_survive_token_position_changes_and_reject_stale_batch(self):
        a = TokenBatch((RequestTokens('a', (11, 12), 0, 'verify'),))
        b = TokenBatch((RequestTokens('b', (21, 22), 7, 'verify'),))
        inputs = PackedInputBuffers(a, 'cpu')
        pointers = tuple(t.data_ptr() for t in (inputs.ids, inputs.positions, inputs.logits_indices))
        inputs.update(b)
        self.assertEqual(pointers, tuple(t.data_ptr() for t in (inputs.ids, inputs.positions, inputs.logits_indices)))
        self.assertEqual(inputs.ids.tolist(), [21, 22])
        self.assertEqual(inputs.positions.tolist(), [7, 8])
        with self.assertRaises(ValueError):
            inputs.validate(a)
        with self.assertRaises(ValueError):
            inputs.update(TokenBatch((RequestTokens('b', (21,), 7, 'decode'),)))

    def test_advancing_and_rejecting_fixed_bindings_matches_variable_attention(self):
        for window in (None, 3):
            normal = PackedKVSession({'layer': (1, 4, 3)}, 32, device='cpu')
            bound = BoundKVSession(PackedKVSession({'layer': (1, 4, 3)}, 32, device='cpu'), (1, 4), 16)
            impl = SimpleNamespace(num_heads=2, num_kv_heads=1, head_size=4, head_size_v=3,
                scale=.5, sliding_window=window, sinks=torch.tensor([.2, -.1]),
                kv_sharing_target_layer_name=None, alibi_slopes=None)
            torch.manual_seed(23)
            pointers, previous = None, None
            for step, commits in enumerate(((1, 2), (1, 0), (1, 4), (1, 1))):
                starts = tuple(len(normal.committed.get(r, ())) for r in ('a', 'b'))
                batch = TokenBatch((RequestTokens('a', (11 + step,), starts[0], 'decode'),
                                    RequestTokens('b', (21, 22, 23, 24), starts[1], 'verify')))
                a, b = normal.prepare(batch), bound.prepare(batch)
                if previous is not None:
                    with self.assertRaises(ValueError):
                        bound.commit(previous, (1, 1))
                current = tuple(t.data_ptr() for t in (b.slot_mapping, *b.visible_slots,
                                  *b.visible_lengths, *b.query_positions))
                self.assertEqual(current, pointers or current)
                pointers = current
                q, k, v = torch.randn(5, 2, 4), torch.randn(5, 1, 4), torch.randn(5, 1, 3)
                x = forward_packed(impl, SimpleNamespace(layer_name='layer'), q, k, v, a)
                y = forward_packed(impl, SimpleNamespace(layer_name='layer'), q, k, v, b)
                torch.testing.assert_close(y, x, rtol=2e-6, atol=2e-6)
                normal.commit(a, commits)
                bound.commit(b, commits)
                self.assertEqual(normal.committed, bound.arena.committed)
                previous = b

    def test_capacity_failure_does_not_reserve_slots(self):
        arena = PackedKVSession({'layer': (1, 4, 3)}, 8, device='cpu')
        bound = BoundKVSession(arena, (2,), 2)
        batch = TokenBatch((RequestTokens('a', (1, 2), 0, 'verify'),))
        meta = bound.prepare(batch)
        bound.commit(meta, (1,))
        with self.assertRaises(ValueError):
            bound.prepare(TokenBatch((RequestTokens('a', (3, 4), 1, 'verify'),)))
        self.assertIsNone(arena.pending)
        self.assertIsNone(bound.pending)
