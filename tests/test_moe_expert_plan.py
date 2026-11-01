import random
import unittest
from gaudi_kernels.moe_expert_plan import ExpertPlan,ExpertBucket,reference_partition


class ExpertPlanTests(unittest.TestCase):
    def test_joint_capacity_budgets_bound_padded_work_not_average_m(self):
        plan=ExpertPlan(8,row_caps=(16,64,512),total_slot_budget=256,total_padded_row_budget=6144)
        cost=plan.costs(512,8,384)
        self.assertLessEqual(cost['slots'],256)
        self.assertLessEqual(cost['padded_mme_rows'],6144)
        self.assertTrue(all(b.fallback_overflow for b in cost['buckets']))
        self.assertTrue(all(b.selection=='largest_m' for b in cost['buckets']))
        self.assertEqual(ExpertPlan(8,total_padded_row_budget=1).buckets(512,8,384),())
        # Measured broadcast scratch alone is201,326,848 bytes for this
        # HistoricalN512 geometry. A masked fallback does not shrink it.
        self.assertGreaterEqual(cost['tpc_fallback_bound_bytes'],201326848)
        self.assertGreaterEqual(cost['temporary_bound_bytes'],251658496)

    def test_global_slots_preserve_every_route_under_changing_skew(self):
        for t in (32, 512, 2048):
            states = [
                [[(i*8+j)%384 for j in range(8)] for i in range(t)],
                [list(range(376,384)) for _ in range(t)],
                [[0,1,2,3,*[4+(i*4+j)%380 for j in range(4)]] for i in range(t)],
            ]
            for budget in (1, 8, 32):
                plan = ExpertPlan(8, total_slot_budget=budget)
                buckets = plan.buckets(t,8,384)
                self.assertLessEqual(sum(b.slots for b in buckets),budget)
                for ids in states:
                    hits=[0]*(t*8)
                    for bucket in buckets:
                        ref=reference_partition(ids,384,bucket)
                        self.assertEqual(ref['status'],[0])
                        for q,row in enumerate(ref['inverse']):
                            hits[q] += row >= 0
                            if row >= 0:
                                self.assertEqual(ref['row_map'][row],q)
                    self.assertTrue(all(hit in (0,1) for hit in hits))
                    # Complement is precisely the TPC-owned route set, never
                    # a second MME owner or a route dropped on overflow.
                    self.assertEqual(sum(hits)+sum(hit==0 for hit in hits),t*8)

    def test_capacity_selects_actual_m_then_stable_ties(self):
        ids=[[0,4],[1,4],[2,4],[3,4],[3,4],[2,4]]
        bucket=ExpertBucket(1,6,6,2,True,'largest_m')
        ref=reference_partition(ids,5,bucket)
        self.assertEqual(ref['tile_expert'],[2,4])
        self.assertEqual(ref['valid_rows'],[2,6])
        self.assertEqual(ref['status'],[0])
        rotated=[[e if e!=4 else 1 for e in (3,4)] for _ in range(6)]
        ref=reference_partition(rotated,5,bucket)
        self.assertEqual(ref['tile_expert'],[1,3])

    def test_unique_expert_bucket_and_route_coverage(self):
        random.seed(819)
        for t,e,r in ((1,8,8),(8,20,8),(33,32,8),(128,64,8),(512,384,8),(513,8,8)):
            ids=[random.sample(range(e),r) for _ in range(t)]
            for threshold in (1,3,8,17,64,129,514):
                buckets=ExpertPlan(threshold).buckets(t,r,e)
                counts=[sum(row.count(ex) for row in ids) for ex in range(e)]
                hits=[0]*(t*r);experts_seen=[]
                for b in buckets:
                    ref=reference_partition(ids,e,b)
                    self.assertEqual(ref['status'],[0])
                    selected=[x for x in ref['tile_expert'] if x>=0]
                    experts_seen+=selected
                    self.assertTrue(all(b.lower<=counts[x]<=b.upper<=b.rows for x in selected))
                    for q,row in enumerate(ref['inverse']):
                        if row>=0:
                            self.assertEqual(ref['row_map'][row],q);hits[q]+=1
                self.assertEqual(len(experts_seen),len(set(experts_seen)))
                for q,eid in enumerate(x for row in ids for x in row):
                    self.assertEqual(hits[q],int(counts[eid]>=threshold))

    def test_mixed_replays_and_status_not_silently_masked(self):
        b=ExpertBucket(2,4,4,4)
        for ids in ([[0,1],[0,2],[0,3],[1,2]],[[0,1],[2,3],[4,5],[6,7]]):
            ref=reference_partition(ids,8,b)
            self.assertEqual(ref['status'],[0])
            for q,row in enumerate(ref['inverse']):
                e=ids[q//2][q%2]
                self.assertEqual(row>=0,ref['counts'][e]>=2)
        for ids in ([[0,0],[1,2]],[[0,-1],[1,2]]):
            ref=reference_partition(ids,8,b)
            self.assertNotEqual(ref['status'],[0]);self.assertEqual(ref['inverse'],[-1]*4)
        self.assertNotEqual(reference_partition([[0,1],[0,1]],8,ExpertBucket(2,2,2,1))['status'],[0])

    def test_no_fake_measured_threshold_or_unbounded_budget(self):
        self.assertEqual(ExpertPlan(64).buckets(4,8,384),())
        self.assertFalse(ExpertPlan(1,workspace_budget=1).costs(512,8,384)['fits_budget'])
        for t in (-1,4097,True):
            with self.assertRaises(ValueError):ExpertPlan().buckets(t,8,384)

    def test_bounded_capacity_falls_back_without_losing_routes(self):
        rng=random.Random(92)
        for t in (32,128,512):
            ids=[rng.sample(range(32),8) for _ in range(t)]
            for cap in (1,4,8):
                plan=ExpertPlan(2,max_slots_per_bucket=cap,row_caps=(t,))
                bucket,=plan.buckets(t,8,32)
                ref=reference_partition(ids,32,bucket)
                self.assertEqual(ref['status'],[0])
                selected=set(x for x in ref['tile_expert'] if x>=0)
                self.assertLessEqual(len(selected),cap)
                for q,e in enumerate(x for row in ids for x in row):
                    self.assertEqual(ref['inverse'][q]>=0,e in selected)
                self.assertEqual(sum(ref['valid_rows']),sum(n for e,n in enumerate(ref['counts']) if e in selected))

    def test_large_chunks_have_unique_route_ownership_and_bounded_workspace(self):
        from gaudi_kernels.serving.executor.grouped_moe_runtime import plan_for
        p512 = plan_for(512)
        self.assertEqual(tuple((b.lower, b.upper, b.rows, b.slots) for b in p512.buckets(512, 8, 384)),
                         ((1, 64, 64, 384), (65, 512, 512, 63)))
        for ids in (
            [[j for j in range(8)] for _ in range(512)],
            [[(i // 16) * 8 + j for j in range(8)] for i in range(512)],
            [[(i*8+j)%384 for j in range(8)] for i in range(512)],
            [[376+j if i<16 else (i*8+j)%376 for j in range(8)] for i in range(512)],
        ):
            hits = [0]*(512*8)
            for bucket in p512.buckets(512, 8, 384):
                ref = reference_partition(ids, 384, bucket)
                self.assertEqual(ref['status'], [0])
                for q, row in enumerate(ref['inverse']):
                    hits[q] += row >= 0
            self.assertEqual(set(hits), {1})
        for t in (16, 32, 64, 100, 128, 256, 300, 384, 505, 512, 513, 768, 1000, 1024, 1536, 2048, 3072, 4096):
            ids = [[0,1,2,3,*[4+(i*4+j)%380 for j in range(4)]] for i in range(t)]
            plan = plan_for(t)
            self.assertTrue(plan.costs(t,8,384)['fits_budget'])
            hits = [0]*(t*8)
            for bucket in plan.buckets(t,8,384):
                ref = reference_partition(ids,384,bucket)
                self.assertEqual(ref['status'],[0])
                for q,row in enumerate(ref['inverse']):
                    hits[q] += row>=0
                    if row>=0:self.assertEqual(ref['row_map'][row],q)
            self.assertEqual(set(hits),{1})
