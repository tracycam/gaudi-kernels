import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from gaudi_kernels.serving.graph_options import GraphOptions


class GraphOptionsTest(unittest.TestCase):
    def test_no_manifest_or_model_configuration_is_required(self):
        self.assertEqual(GraphOptions.from_vllm(SimpleNamespace()), GraphOptions())
        config = SimpleNamespace(additional_config={'unrelated': 3, 'hpu_graph': {
            'scope': 'step', 'reuse_inputs': False, 'max_input_buckets': 4}})
        self.assertEqual(GraphOptions.from_vllm(config), GraphOptions('step', False, 4))

    def test_unknown_keys_and_coercions_reject_before_execution(self):
        for value in ('step', {'scope': 'native'}, {'reuse_inputs': 'false'},
                      {'max_input_buckets': True}, {'max_input_buckets': 0}, {'surprise': True},
                      {'max_cached_graphs': True}, {'max_cached_graphs': 0}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GraphOptions.from_vllm(SimpleNamespace(additional_config={'hpu_graph': value}))

    def test_capture_limit_is_distinct_from_prepared_input_capacity(self):
        value = GraphOptions.from_vllm(SimpleNamespace(additional_config={'hpu_graph': {
            'max_cached_graphs': 32, 'max_input_buckets': 4}}))
        self.assertEqual(value.max_cached_graphs, 32)
        self.assertEqual(value.max_input_buckets, 4)

    def test_generic_runner_has_no_model_manifest_or_operator_policy_import(self):
        root = Path(__file__).resolve().parents[1]/'python/gaudi_kernels/serving'
        tree = ast.parse((root/'runner.py').read_text())
        modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertNotIn('gaudi_kernels.engine.context', modules)
        self.assertNotIn('gaudi_kernels.serving.policy', modules)
        self.assertNotIn('gaudi_kernels.serving.model_adapter', modules)

    def test_scope_uses_shared_application_options_without_model_fields(self):
        source = Path(__file__).resolve().parents[1]/'python/gaudi_kernels/serving/runner.py'
        cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('graph_options', '_step_graph_enabled')]
        namespace = {'GraphOptions': GraphOptions}
        tree = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Runner', bases=[], keywords=[],
             body=methods, decorator_list=[])], type_ignores=[]))
        exec(compile(tree, str(source), 'exec'), namespace)
        runner = namespace['Runner']()
        runner.vllm_config = SimpleNamespace(additional_config={'hpu_graph': {'scope': 'step'}})
        self.assertTrue(runner._step_graph_enabled())
        self.assertIs(runner.graph_options(), runner.graph_options())
        runner._step_graph_override = False
        self.assertFalse(runner._step_graph_enabled())

    def test_generic_prefill_preserves_the_original_graph_policy(self):
        source = Path(__file__).resolve().parents[1]/'python/gaudi_kernels/serving/runner.py'
        cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_use_graphs')
        class Base:
            def _use_graphs(self, *args):
                return True
        tree = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Runner', bases=[ast.Name(id='Base', ctx=ast.Load())],
               keywords=[], body=[method], decorator_list=[])], type_ignores=[]))
        namespace = {'Base': Base}
        exec(compile(tree, str(source), 'exec'), namespace)
        runner = namespace['Runner']()
        self.assertTrue(runner._use_graphs(SimpleNamespace(is_prompt=True), 2))
        runner._inside_step_graph = True
        self.assertFalse(runner._use_graphs(SimpleNamespace(is_prompt=False), 2))
