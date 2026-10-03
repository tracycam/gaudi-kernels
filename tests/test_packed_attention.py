import math
from types import SimpleNamespace
import unittest

import torch

from gaudi_kernels.engine.token_batch import RequestTokens, TokenBatch
from gaudi_kernels.serving.executor.packed_attention import PackedKVSession, forward_packed, tiled_attention


class PackedAttentionTests(unittest.TestCase):
    def test_diff_kv_gqa_sinks_and_swa_match_dense(self):
        torch.manual_seed(7)
        q, k, v = (torch.randn(4, 4, 8), torch.randn(301, 2, 8), torch.randn(301, 2, 6))
        query_positions = torch.arange(297, 301)
        key_positions = torch.arange(301)
        for window in (None, 128):
            for sinks in (None, torch.tensor([.2, -.1, .5, .9])):
                for tile in (1, 31, 128, 256):
                    result = tiled_attention(q, k, v, query_positions, key_positions, scale=1/math.sqrt(8),
                                             sliding_window=window, sinks=sinks, tile_size=tile)
                    score = q.transpose(0, 1) @ k.repeat_interleave(2, 1).permute(1, 2, 0) / math.sqrt(8)
                    allowed = key_positions[None] <= query_positions[:, None]
                    if window:
                        allowed &= key_positions[None] > query_positions[:, None] - window
                    score = score.masked_fill(~allowed, float('-inf'))
                    if sinks is not None:
                        score = torch.cat((score, sinks[:, None, None].expand(4, 4, 1)), -1)
                    expected = torch.softmax(score, -1)[..., :301] @ v.repeat_interleave(2, 1).transpose(0, 1)
                    torch.testing.assert_close(result, expected.transpose(0, 1), rtol=3e-5, atol=3e-6)

    def test_rejection_reuses_only_dead_slots_and_keeps_visible_kv(self):
        arena = PackedKVSession({'layer': (1, 4, 3)}, 6, device='cpu')
        impl = SimpleNamespace(num_heads=2, num_kv_heads=1, head_size=4, head_size_v=3,
                               scale=.5, sliding_window=2, sinks=None,
                               kv_sharing_target_layer_name=None, alibi_slopes=None)
        layer = SimpleNamespace(layer_name='layer')
        initial = arena.prepare(TokenBatch((RequestTokens('a', (1, 2), 0, 'prefill'),)))
        q, k, v = torch.ones(2, 2, 4).bfloat16(), torch.ones(2, 1, 4).bfloat16(), torch.ones(2, 1, 3).bfloat16()
        forward_packed(impl, layer, q, k, v, initial)
        arena.commit(initial, (2,))
        before = tuple(t[:2].clone() for t in arena.caches['layer'])
        pointers = tuple(t.data_ptr() for t in arena.caches['layer'])
        candidate = arena.prepare(TokenBatch((RequestTokens('a', (3, 4, 5), 2, 'verify'),)))
        forward_packed(impl, layer, q[:1].expand(3, -1, -1), k[:1].expand(3, -1, -1)*2,
                       v[:1].expand(3, -1, -1)*2, candidate)
        with self.assertRaises(ValueError):
            arena.commit(candidate, (4,))
        self.assertIs(arena.pending, candidate)
        arena.commit(candidate, (1,))
        self.assertEqual(tuple(t.data_ptr() for t in arena.caches['layer']), pointers)
        following = arena.prepare(TokenBatch((RequestTokens('a', (6,), 3, 'decode'),)))
        self.assertEqual(arena.committed['a'], (0, 1, 2))
        self.assertEqual(following.candidate_slots, ((3,),))
        for actual, expected in zip(arena.caches['layer'], before):
            self.assertTrue(torch.equal(actual[:2], expected))
        arena.abort(following)
        with self.assertRaises(ValueError):
            arena.commit(candidate, (1,))

    def test_request_isolation_and_future_causality(self):
        def run(changed):
            arena = PackedKVSession({'layer': (1, 4, 3)}, 8, device='cpu')
            batch = TokenBatch((RequestTokens('a', (1,), 0, 'decode'), RequestTokens('b', (2, 3), 0, 'verify')))
            metadata = arena.prepare(batch)
            impl = SimpleNamespace(num_heads=2, num_kv_heads=1, head_size=4, head_size_v=3,
                                   scale=.5, sliding_window=None, sinks=None,
                                   kv_sharing_target_layer_name=None, alibi_slopes=None)
            torch.manual_seed(19)
            q, k, v = torch.randn(3, 2, 4), torch.randn(3, 1, 4), torch.randn(3, 1, 3)
            if changed:
                k[2] *= -10
                v[2] *= -10
            return forward_packed(impl, SimpleNamespace(layer_name='layer'), q, k, v, metadata)
        a, b = run(False), run(True)
        self.assertTrue(torch.equal(a[:2], b[:2]))
        self.assertFalse(torch.equal(a[2], b[2]))
