import random
import unittest
from gaudi_kernels.moe_expert_plan import ExpertPlan,ExpertBucket,reference_partition


class ExpertPlanTests(unittest.TestCase):
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
        for t in (-1,514,True):
            with self.assertRaises(ValueError):ExpertPlan().buckets(t,8,384)
