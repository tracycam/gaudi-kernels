# SPDX-License-Identifier: Apache-2.0
import unittest

from gaudi_kernels.engine.token_batch import RequestTokens, TokenBatch
from gaudi_kernels.serving.executor.packed_bindings import BoundKVSession
from gaudi_kernels.serving.executor.packed_pages import PagedKVSession


class PackedPageTests(unittest.TestCase):
    def test_private_pages_cover_query_window_and_rejection_without_live_alias(self):
        arena = PagedKVSession({'layer': (1, 4, 3)}, {'a': 260, 'b': 260}, device='cpu')
        initial = TokenBatch(tuple(RequestTokens(r, tuple(range(127)), 0, 'prefill') for r in ('a', 'b')))
        meta = arena.prepare(initial)
        arena.commit(meta, (127, 127))
        live = {s for slots in arena.committed.values() for s in slots}
        bound = BoundKVSession(arena, (3, 3), 260)
        batch = TokenBatch(tuple(RequestTokens(r, (1, 2, 3), 127, 'verify') for r in ('a', 'b')))
        candidate = bound.prepare(batch)
        self.assertEqual(candidate.window_page_ids.tolist(), [0, -1, 0, 1, 0, 1, 3, -1, 3, 4, 3, 4])
        self.assertEqual(candidate.window_page_groups.tolist(), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        self.assertEqual(candidate.flat_query_positions.tolist(), [127, 128, 129, 127, 128, 129])
        self.assertFalse(live.intersection(candidate.slot_mapping.tolist()))
        addresses = tuple(t.data_ptr() for t in (candidate.window_page_ids,
                         candidate.window_page_groups, candidate.flat_query_positions))
        bound.commit(candidate, (1, 2))
        next_batch = TokenBatch((RequestTokens('a', (4, 5, 6), 128, 'verify'),
                                RequestTokens('b', (4, 5, 6), 129, 'verify')))
        following = bound.prepare(next_batch)
        self.assertEqual(addresses, tuple(t.data_ptr() for t in (following.window_page_ids,
                                          following.window_page_groups, following.flat_query_positions)))
        self.assertEqual(following.candidate_slots, ((128, 129, 130), (513, 514, 515)))
        self.assertFalse({s for slots in arena.committed.values() for s in slots}.intersection(following.slot_mapping.tolist()))
        with self.assertRaises(ValueError):
            bound.commit(candidate, (1, 1))
        bound.abort(following)

    def test_unknown_request_and_extent_fail_before_reservation(self):
        arena = PagedKVSession({'layer': (1, 4, 3)}, {'a': 2}, device='cpu')
        for request in (RequestTokens('unknown', (1,), 0, 'decode'),
                        RequestTokens('a', (1, 2, 3), 0, 'verify')):
            with self.assertRaises(ValueError):
                arena.prepare(TokenBatch((request,)))
            self.assertIsNone(arena.pending)
            self.assertEqual(arena.committed, {})
