import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from gaudi_kernels.engine import ConfigError, DispatchTable, EngineConfig, Request, load_config
from tools.validation.legacy_selection import PolicySelection


class ConfigurationTests(unittest.TestCase):
    def test_saved_70g_selection(self):
        # This exact strategy is the independently saved 70-g ABBA control,
        # not a spelling reconstructed from this implementation.
        expected = ('decode_a8_bf16_fp32+gp_fold+gp_scale_tail+down_vec+'
            'swa_fp32_av_hoist+qkv_neumaier_isa+qkv_post+f32_ag+'
            'router_post_vector+norm_fp32+moe_compact8+moe_sum_bf16+norm_qkv_grid24')
        self.assertEqual(PolicySelection(EngineConfig()).label, expected)

    def test_framework_execution_rejects_external_replay_flags(self):
        cfg = EngineConfig.from_dict({'runtime': {'executor': 'pytorch'}})
        self.assertEqual(cfg.runtime.executor, 'pytorch')
        cfg = EngineConfig.from_dict({'runtime': {'executor': 'pytorch',
            'buckets': {'batch': [1, 2, 8], 'tokens': [1, 4]}}})
        self.assertEqual(DispatchTable(cfg).runtime_admission(), 'pytorch')
        for setting in ('native_enabled', 'batch_replay'):
            with self.assertRaises(ConfigError):
                EngineConfig.from_dict({'runtime': {'executor': 'pytorch',
                                                    'runner': 'native', setting: True}})

    def test_unknown_fields_and_scalar_types_fail(self):
        for document in ({'surprise': True}, {'decode': {'qkv': {'imlp': 'typo'}}},
                         {'parallel': {'tp': True}}, {'runtime': {'flights': '4'}},
                         {'diagnostics': {'profile': 'false'}},
                         {'runtime': {'buckets': {'batch': [True]}}}):
            with self.subTest(document=document), self.assertRaises(ConfigError):
                EngineConfig.from_dict(document)

    def test_global_mme_limits_are_typed_and_bounded(self):
        for fields in ({'grouped_slot_budget': True}, {'grouped_slot_budget': -1},
                       {'grouped_slot_budget': 1537}, {'grouped_min_mme_rows': 0},
                       {'grouped_min_mme_rows': '8'}, {'grouped_padded_row_budget': True},
                       {'grouped_padded_row_budget': -1}, {'grouped_padded_row_budget': 131073}):
            with self.subTest(fields=fields), self.assertRaises(ConfigError):
                EngineConfig.from_dict({'decode': {'moe': {'dispatch': fields}}})

    def test_duplicate_json_fields_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_invalid_cross_feature_contracts(self):
        for decode in ({'qkv': {'a8_max_rows': 0}},
                       {'attention': {'swa': 'vendor'}},
                       {'qkv': {'post': 'vendor'}},
                       {'tp_reduce': {'collective': 'baseline'}},
                       {'moe': {'dispatch': {'compact_rows': [1, 9]}}}):
            with self.subTest(decode=decode), self.assertRaises(ConfigError):
                EngineConfig.from_dict({'decode': decode})

    def test_bucket_and_flight_validation(self):
        for runtime in ({'flights': 0}, {'flights': 17},
                        {'buckets': {'batch': []}}, {'buckets': {'tokens': [2, 1]}},
                        {'buckets': {'batch': [1, 1]}}):
            with self.subTest(runtime=runtime), self.assertRaises(ConfigError):
                EngineConfig.from_dict({'runtime': runtime})

    def test_config_is_detached_and_round_trips(self):
        document = EngineConfig().to_dict()
        parsed = EngineConfig.from_dict(document)
        document['decode']['moe']['dispatch']['compact_rows'].append(999)
        self.assertEqual(parsed, EngineConfig())
        self.assertEqual(EngineConfig.from_dict(parsed.to_dict()), parsed)

    def test_diagnostics_do_not_change_arithmetic_fingerprint(self):
        baseline = EngineConfig()
        diagnostic = EngineConfig.from_dict({'diagnostics': {'profile': True, 'layer_hashes': True}})
        self.assertEqual(baseline.arithmetic_fingerprint, diagnostic.arithmetic_fingerprint)
        alternate = EngineConfig.from_dict({'decode': {'qkv': {'post': 'fused_all'}}})
        self.assertNotEqual(baseline.arithmetic_fingerprint, alternate.arithmetic_fingerprint)

    def test_qkv_keeps_qualified_m1_vs_batch_split(self):
        table = DispatchTable(EngineConfig())
        self.assertEqual(table.resolve(Request('qkv', 1, 3392, 6144)).activation, 'block128_fp8')
        for rows in (2, 4, 8, 16, 512, 513, 1024):
            self.assertEqual(table.resolve(Request('qkv', rows, 3392, 6144)).activation, 'bf16')
        self.assertEqual(table.resolve(Request('qkv', 1, 4096, 6144)).activation, 'bf16')
        with self.assertRaises(ValueError):
            table.resolve(Request('qkv', 1))

    def test_explicit_multirow_policy_matches_existing_backend_row_contract(self):
        cfg = EngineConfig.from_dict({'decode':{'qkv':{'a8_max_rows':16},
            'attention':{'swa':'fp32_batch_quad'},
            'tp_reduce':{'fp32_max_rows':16,'fp32_max_bytes':524288}}})
        from gaudi_kernels.engine.selection import PolicySelection as TypedSelection
        self.assertEqual(TypedSelection(cfg).block_policy,'small_batch_a8_bf16_fp32')
        table = DispatchTable(cfg)
        for rows in (1,2,3,4,8,12,16):
            self.assertEqual(table.resolve(Request('qkv',rows,3392,6144)).activation,'block128_fp8')
        self.assertEqual(table.resolve(Request('qkv',17,3392,6144)).activation,'bf16')
        self.assertEqual(cfg.decode.qkv.post_max_rows,1)
        self.assertEqual(EngineConfig.from_dict({'decode':{'qkv':{'post_max_rows':16}}}).decode.qkv.post_max_rows,16)
        self.assertEqual(EngineConfig.from_dict({'decode':{'qkv':{'post_max_rows':4096}}}).decode.qkv.post_max_rows,4096)
        with self.assertRaises(ConfigError):
            EngineConfig.from_dict({'decode':{'qkv':{'a8_max_rows':17}}})
        with self.assertRaises(ConfigError):
            EngineConfig.from_dict({'decode':{'qkv':{'post_max_rows':32768}}})

    def test_compact_row_set_can_preserve_b2_broadcast(self):
        config = EngineConfig.from_dict({'decode': {'moe': {'dispatch': {'compact_rows': [1, 8]}}}})
        table = DispatchTable(config)
        for rows in (1, 8):
            self.assertEqual(table.resolve(Request('moe_dispatch', rows)).name, 'mxfp4.compact')
        for rows in (2, 3, 4, 16, 512):
            self.assertEqual(table.resolve(Request('moe_dispatch', rows)).name, 'mxfp4.broadcast')

    def test_grouped_shapes_are_explicit_and_preserve_small_batch(self):
        cfg = EngineConfig.from_dict({'decode': {'moe': {'dispatch': {
            'grouped_rows': [512, 2048, 4096]}}}})
        table = DispatchTable(cfg)
        for rows in (1, 2, 8):
            self.assertEqual(table.resolve(Request('moe_dispatch', rows)).name, 'mxfp4.compact')
        for rows in (16, 513, 1024):
            self.assertEqual(table.resolve(Request('moe_dispatch', rows)).name, 'mxfp4.broadcast')
        for rows in (512, 2048, 4096):
            self.assertEqual(table.resolve(Request('moe_dispatch', rows)).name, 'mxfp4.grouped')
        for rows in ([1], [512, 512], [32768], [True]):
            with self.assertRaises(ConfigError):
                EngineConfig.from_dict({'decode': {'moe': {'dispatch': {'grouped_rows': rows}}}})

    def test_unqualified_executor_or_native_batch_refused(self):
        self.assertEqual(DispatchTable(EngineConfig()).runtime_admission(), 'pytorch')
        for runtime in ({'executor': 'native_graph'}, {'executor': 'legacy_native'},
                        {'runner': 'legacy_hooks'}):
            with self.assertRaises(ConfigError):
                EngineConfig.from_dict({'runtime': runtime})

    def test_named_batch_requires_owned_runner_and_cannot_enable_verify(self):
        for runtime in ({'native_enabled': True}, {'batch_replay': True},
                        {'runner':'native', 'batch_replay':True, 'buckets':{'tokens':[1,4]}}):
            with self.assertRaises(ConfigError):
                EngineConfig.from_dict({'runtime':runtime})
        cfg = EngineConfig.from_dict({'runtime':{'runner':'native',
                                                'buckets':{'batch':[1,2,3]}}})
        self.assertEqual(DispatchTable(cfg).runtime_admission(), 'pytorch')

    def test_workers_refuse_strategy_strings(self):
        with self.assertRaises(ConfigError):
            PolicySelection.from_dict(PolicySelection(EngineConfig()).label)
        with self.assertRaises(ConfigError):
            PolicySelection.from_dict({'engine': {}, 'reference': 'fp64_exact'})
        with self.assertRaises(ConfigError):
            PolicySelection(EngineConfig.from_dict({'decode': {'moe': {'dispatch': {'compact_rows': [1, 8]}}}})).features

    def test_closed_lookup_covers_saved_70g_quality_references(self):
        path = Path(__file__).resolve().parents[1] / 'tools/validation/legacy_selection.py'
        spec = importlib.util.spec_from_file_location('migration', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        table = module.qualified_selections()
        for impl in ('cpu_ocp_a8_fp32_v1', 'cpu_native_a8_fp32_v1'):
            label = impl + '+norm_fp32+swa_fp32_fast+f32_ag'
            selected = PolicySelection.from_dict(table[label])
            self.assertEqual(selected.reference, impl)
            self.assertEqual(selected.engine.decode.attention.swa, 'fp32_fast')


if __name__ == '__main__':
    unittest.main()
