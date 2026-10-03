from dataclasses import replace
import unittest
from gaudi_kernels.engine.kv_plan import KVPageView, KVStagingPlan
from gaudi_kernels.engine.token_batch import RequestTokens, TokenBatch


class KVPlanTests(unittest.TestCase):
    def test_swa_wrap_candidates_never_overwrite_visible_ring_and_commit_only_prefix(self):
        view = KVPageView('pool-generation-1', ('r',), (128,), (tuple(range(128)),), 1, 136, 128)
        batch = TokenBatch((RequestTokens('r', (7, 8, 9, 10), 128, 'verify'),))
        with self.assertRaisesRegex(ValueError, 'alias live'):
            KVStagingPlan(view, batch, ((0, 1, 2, 3),))
        plan = KVStagingPlan(view, batch, ((128, 129, 130, 131),))
        for count in range(5):
            selected = plan.select_commit((count,), current_view=view)
            self.assertEqual(selected.requests[0].new_length, 128 + count)
            self.assertEqual(selected.requests[0].promotions, tuple((128 + i, 128 + i) for i in range(count)))
            self.assertEqual(selected.discarded_slots, tuple(range(128 + count, 132)))
        with self.assertRaisesRegex(ValueError, 'Stale'):
            plan.select_commit((2,), current_view=replace(view, lease_id='pool-generation-2'))

    def test_shared_prefix_pages_are_read_only_and_candidate_slots_are_unique(self):
        view = KVPageView('shared-pool', ('a', 'b'), (3, 3), ((0,), (0,)), 4, 4)
        batch = TokenBatch((RequestTokens('a', (11,), 3, 'decode'),
                            RequestTokens('b', (12,), 3, 'decode')))
        with self.assertRaisesRegex(ValueError, 'alias live'):
            KVStagingPlan(view, batch, ((2,), (4,)))
        with self.assertRaisesRegex(ValueError, 'another query'):
            KVStagingPlan(view, batch, ((3,), (3,)))
        plan = KVStagingPlan(view, batch, ((3,), (4,)))
        self.assertEqual(plan.select_commit((1, 0), current_view=view).discarded_slots, (4,))

    def test_partial_page_visibility_protects_only_the_committed_positions(self):
        view = KVPageView('pages', ('r',), (129,), ((1, 2),), 128, 4, 128)
        self.assertEqual(view.visible_ranges(0), ((129, 256), (256, 257)))
        batch = TokenBatch((RequestTokens('r', (1, 2), 129, 'verify'),))
        with self.assertRaisesRegex(ValueError, 'alias live'):
            KVStagingPlan(view, batch, ((256, 300),))
        self.assertEqual(KVStagingPlan(view, batch, ((257, 258),)).staging_slots, ((257, 258),))

    def test_invalid_coverage_cursor_and_commit_capacity_are_rejected(self):
        with self.assertRaises(ValueError):
            KVPageView('p', ('r',), (129,), ((0,),), 128, 4)
        view = KVPageView('p', ('r',), (0,), ((),), 128, 1)
        with self.assertRaisesRegex(ValueError, 'cursor'):
            KVStagingPlan(view, TokenBatch((RequestTokens('r', (1,), 1, 'decode'),)), ((0,),))
        plan = KVStagingPlan(view, TokenBatch((RequestTokens('r', (1,), 0, 'decode'),)), ((0,),))
        with self.assertRaisesRegex(ValueError, 'prefix'):
            plan.select_commit((2,), current_view=view)
