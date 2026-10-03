import json
from pathlib import Path
import tempfile
import unittest

from tools.validation.executor.summarize_cycle_cohort import summarize


class CycleCohortTests(unittest.TestCase):
    def reports(self, root):
        for rank in range(2):
            cycles = []
            for step, values in enumerate(((1, 9), (8, 2), (90, 90))):
                cycles.append(dict(step=step, pass_=True, starts=[step], emitted=[1],
                    proposed_ids=[[2]], target_next_ids=[[2]], emitted_ids=[[2]], matched_drafts=[1],
                    eager_cycle_wall_ns=values[rank]*1_000_000,
                    execution=dict(kind='recorded-replay', profiled=step == 2,
                        sdk_replay_with_completion_wall_ns=values[rank]*1_000_000)))
                cycles[-1]['pass'] = cycles[-1].pop('pass_')
            (root/f'rank{rank}.json').write_text(json.dumps(dict(rank=rank, pass_=True,
                release_code=0, cycles=cycles)).replace('"pass_":', '"pass":'))

    def test_per_step_max_and_profile_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.reports(root)
            # max(median(rank0), median(rank1)) would be5.5, not8.5.
            self.assertEqual(summarize(root, 2)['median_max_rank_cycle_ms'], 8.5)
            report = json.loads((root/'rank1.json').read_text())
            report['cycles'][0]['emitted_ids'] = [[3]]
            (root/'rank1.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'witness disagreement'):
                summarize(root, 2)
