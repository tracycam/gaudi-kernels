import copy
import unittest

from tools.consolidation.analyze_latency_repeat import analyze, analyze_arm
from tools.consolidation.analyze_runtime import returned_timing


class LatencyAttributionTests(unittest.TestCase):
    clock = {'hostname': 'fixture', 'boot_id': 'fixture-boot', 'implementation': 'monotonic'}

    def row(self):
        row = {'arm': 0, 'cycle': 0, 'kind': 'paired', 'installer': 'historical',
               'ids': list(range(160)), 'execution_gate_pass': True, 'steps': [],
               'ranks': [{'rank': rank, 'clock_domain': self.clock, 'native_steps': []} for rank in range(8)]}
        for i, emitted in enumerate(range(32, 160)):
            begin = i*1_000_000
            row['steps'].append({'begin_ns': begin, 'end_ns': begin+1_000_000,
                                 'ms': 1, 'emitted': emitted, 'new_tokens': 1})
            for rank in row['ranks']:
                start = begin+100_000+rank['rank']*1000
                rank['native_steps'].append({'position': 4096+emitted-2, 'token': emitted-1,
                    'caller_begin_ns': start, 'caller_end_ns': start+600_000, 'caller_ns': 600_000,
                    'native_wall_ns': 550_000, 'enqueue_ns': 300_000,
                    'synlaunch_ns': 100_000, 'hccl_ns': 100_000})
        return row

    def test_common_clock_decomposition_and_rank_skew(self):
        result = analyze_arm(self.row(), context=4096, clock=self.clock)
        self.assertEqual(result['matched_native_steps'], 128)
        first = result['aligned_steps'][0]
        self.assertAlmostEqual(first['rank_start_skew_ms'], .007)
        self.assertAlmostEqual(first['first_caller_begin_offset_ms']+first['caller_envelope_ms']
                               -first['last_caller_end_offset_ms'], first['coordinator_ms'])

    def test_worker_run_ahead_and_late_rank_bookkeeping_are_valid(self):
        row = self.row()
        first = row['ranks'][0]['native_steps'][0]
        first['caller_begin_ns'] -= 200_000
        first['caller_end_ns'] -= 200_000
        last = row['ranks'][7]['native_steps'][0]
        last['caller_begin_ns'] += 400_000
        last['caller_end_ns'] += 400_000
        result = analyze_arm(row, context=4096, clock=self.clock)
        first, second = result['aligned_steps'][:2]
        self.assertLess(first['first_caller_begin_offset_ms'], 0)
        self.assertGreater(first['last_caller_end_offset_ms'], 0)
        self.assertAlmostEqual(second['rank0_return_itl_ms']+second['rank0_to_frontend_delay_change_ms'],
                               second['frontend_return_itl_ms'])

    def test_stall_is_not_assigned_a_native_token_and_remains_in_timing(self):
        row = self.row()
        begin = row['steps'][-1]['end_ns']
        row['steps'].append({'begin_ns': begin, 'end_ns': begin+50_000_000,
                             'ms': 50, 'emitted': 159, 'new_tokens': 0})
        result = analyze_arm(row, context=4096, clock=self.clock)
        self.assertEqual(result['unmatched_steps'], 1)
        self.assertAlmostEqual(result['timing']['tps'], 128/.178)
        self.assertFalse(result['worst_steps'][0]['native_match'])

    def test_real_token_itl_includes_stalls_and_uses_previous_return(self):
        row = self.row()
        row['steps'].insert(0, {'begin_ns': -1_000_000, 'end_ns': 0,
                               'ms': 1, 'emitted': 31, 'new_tokens': 1})
        for step in row['steps'][65:]:
            step['begin_ns'] += 50_000_000
            step['end_ns'] += 50_000_000
        previous = row['steps'][64]
        row['steps'].insert(65, {'begin_ns': previous['end_ns'],
                                'end_ns': previous['end_ns']+50_000_000,
                                'ms': 50, 'emitted': previous['emitted'], 'new_tokens': 0})
        report = returned_timing(row)
        self.assertAlmostEqual(report['tps'], 128/.178)
        self.assertEqual(report['tokens'], 128)

    def test_disabled_api_detail_is_not_reported_as_zero_cost(self):
        row = self.row()
        for rank in row['ranks']:
            for step in rank['native_steps']:
                step.update(api_detail_valid=False, synlaunch_ns=0, hccl_ns=0)
        report = analyze_arm(row, context=4096, clock=self.clock)
        self.assertNotIn('max_hccl_api_ms', report['components'])

    def test_wrong_clock_missing_rank_and_token_mismatch_are_rejected(self):
        for fault in ('clock', 'rank', 'token'):
            with self.subTest(fault=fault):
                row = self.row()
                if fault == 'clock':
                    row['ranks'][0]['clock_domain'] = {**self.clock, 'boot_id': 'other-host'}
                elif fault == 'rank':
                    row['ranks'].pop()
                else:
                    row['ranks'][0]['native_steps'][0]['token'] = 999
                with self.assertRaises(ValueError):
                    analyze_arm(row, context=4096, clock=self.clock)

    def test_sham_noise_never_relaxes_paired_gate(self):
        rows = []
        for arm, installer in enumerate(('historical', 'canonical', 'canonical', 'historical')):
            row = self.row()
            row.update(arm=arm, installer=installer)
            if installer == 'canonical':
                for step in row['steps'][-4:]:
                    step['end_ns'] += 50_000_000
                    step['ms'] += 50
            rows.append(row)
        for arm in range(4):
            row = copy.deepcopy(rows[arm])
            row.update(kind='sham', cycle=1, installer='historical')
            rows.append(row)
        result = analyze({'runs': rows, 'context': 4096, 'clock_domain': self.clock})
        self.assertFalse(result['all_paired_cycles_pass'])
        self.assertGreater(result['sham']['p99_inner_minus_outer_ms'], 0)


if __name__ == '__main__':
    unittest.main()
