import json
from pathlib import Path
import tempfile
import unittest

from tools.consolidation.analyze_device_trace import ENGINES, analyze, duration


class DeviceTraceTests(unittest.TestCase):
    def fixture(self):
        rows = [{'ph': 'M', 'name': 'process_name', 'pid': pid,
                 'args': {'name': '*'+engine+' (accel7)'}} for pid, engine in enumerate(ENGINES)]
        # Two TPC lanes overlap. A MME operation overlaps TPC; DMA covers a gap.
        for pid, tid, begin, end in ((0, 0, 0, 1000), (0, 1, 500, 1500),
                                     (1, 0, 1250, 1750), (3, 0, 2000, 2250)):
            rows += [{'ph': 'B', 'pid': pid, 'tid': tid, 'id': tid, 'ts': begin},
                     {'ph': 'E', 'pid': pid, 'tid': tid, 'id': tid, 'ts': end}]
        return rows

    def check(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'trace.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            return analyze(path, 1)

    def test_wall_time_unions_not_lane_sums(self):
        result = self.check(self.fixture())['average_per_replay_ms']
        self.assertEqual(result['TPC'], 1.5)
        self.assertEqual(result['compute_union'], 1.75)
        self.assertEqual(result['TPC_MME_overlap'], .25)
        self.assertEqual(result['selected_engine_idle'], .25)
        self.assertEqual(result['compute_idle'], .5)

    def test_incomplete_capture_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unclosed'):
            self.check(self.fixture()[:-1])
        with self.assertRaisesRegex(ValueError, 'End without'):
            self.check(self.fixture()[0:5]+self.fixture()[6:])

    def test_gap_accounting_includes_compute_edges(self):
        result = self.check(self.fixture())
        idle = sum(row['average_total_ms_per_replay'] for row in result['compute_gap_bins'].values())
        idle += result['compute_edge_idle_ms_per_replay']
        self.assertEqual(idle, result['average_per_replay_ms']['compute_idle'])
        self.assertEqual(result['compute_edge_idle_ms_per_replay'], .5)

    def test_union_handles_touching_and_empty_intervals(self):
        self.assertEqual(duration([]), 0)
        self.assertEqual(duration([(1, 4), (2, 3), (4, 5)]), 4)


if __name__ == '__main__':
    unittest.main()
