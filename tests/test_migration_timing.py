import unittest

from tools.consolidation.compare_migration import timing


class MigrationTimingTests(unittest.TestCase):
    def row(self, stalls=False):
        steps=[]
        cursor=0
        for token in range(32, 160):
            values=[(1, 1)]
            if stalls and token==64:
                values += [(0, 500), (0, 600)]
            for produced, ms in values:
                end=cursor+ms*1_000_000
                steps.append(dict(emitted=token,new_tokens=produced,ms=ms,
                                  begin_ns=cursor,end_ns=end))
                cursor=end
        return {'steps':steps}

    def test_stalls_affect_the_full_interval_and_latency_distribution(self):
        result=timing([self.row(stalls=True)])
        self.assertEqual(result['steps'],130)
        self.assertAlmostEqual(result['tps'],128/1.228)
        self.assertGreater(result['p99_ms'],1)

    def test_incomplete_returned_tokens_cannot_qualify(self):
        row=self.row()
        row['steps'].pop()
        with self.assertRaises(ValueError):timing([row])


if __name__=='__main__':unittest.main()
