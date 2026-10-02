import importlib.util
from pathlib import Path
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'tools/consolidation/compare_migration.py'
SPEC = importlib.util.spec_from_file_location('migration_acceptance', MODULE)
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


class MigrationAcceptanceTests(unittest.TestCase):
    def test_no_audit_is_not_a_pass(self):
        with self.assertRaises(ValueError):
            acceptance.layer_hashes({'quality_plan': {'candidate_policies': ['candidate']},
                                     'teacher_forced': {'rows': []}})

    def test_full_layer_grid_is_required(self):
        def result(omit=None):
            rows = []
            for position in (0, 16, 48):
                ranks = []
                for rank in range(8):
                    records = []
                    for layer in range(70):
                        if (position, rank, layer) == omit:
                            continue
                        records.append({'layer': f'language_model.model.layers.{layer}.self_attn.qkv_proj',
                            'input_sha256': 'a'*64, 'plain_vs_staged': {'plain_sha256': 'b'*64,
                                'staged_sha256': 'b'*64, 'all_bits_equal': True},
                            'fp32_contract': {'numerical_passed': True, 'stages': {name: {'actual_sha256': 'c'*64}
                                for name in ('q_native', 'activation_scales', 'prepared_weight', 'prepared_scales',
                                             'bias', 'partial', 'epilogue')}}})
                    ranks.append({'rank': rank, 'same_input_qkv_audit': records})
                rows.append({'policy': 'candidate', 'position': position, 'same_input_qkv_audit': ranks})
            return {'quality_plan': {'candidate_policies': ['candidate']}, 'teacher_forced': {'rows': rows}}
        self.assertEqual(len(acceptance.layer_hashes(result())), 1680)
        for omitted in ((0, 0, 0), (16, 5, 69), (48, 7, 69)):
            with self.subTest(omitted=omitted), self.assertRaises(ValueError):
                acceptance.layer_hashes(result(omitted))

    def test_incomplete_timing_window_is_rejected(self):
        with self.assertRaises(ValueError):
            acceptance.timing([{'steps': []}])


if __name__ == '__main__':
    unittest.main()
