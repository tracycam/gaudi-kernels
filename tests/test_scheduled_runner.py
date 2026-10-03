"""Exercise owned runner admission/packing at its real override boundary."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class ScheduledRunnerTests(unittest.TestCase):
    def test_real_prepare_override_observes_compact_work_without_changing_execution(self):
        source = Path(__file__).resolve().parents[1] / 'python/gaudi_kernels/serving/runner.py'
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))

        class Base:
            calls = 0

            def _prepare_inputs(self, schedule, prefills, decodes, warmup=False):
                self.calls += 1
                return (prefills, decodes, warmup)

        namespace = {'hpu': SimpleNamespace(HPUModelRunner=Base)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
        runner = namespace['NativeHPUModelRunner']()
        runner._native_transfers = SimpleNamespace(audit_inputs={})
        runner.input_batch = SimpleNamespace(num_reqs=2, req_ids=['a', 'b'], num_computed_tokens_cpu=[0, 0],
            num_prompt_tokens=[4, 128], token_ids_cpu=[[1, 2, 3, 4], list(range(128))])
        runner.requests = {'a': SimpleNamespace(num_tokens=4), 'b': SimpleNamespace(num_tokens=128)}
        schedule = SimpleNamespace(num_scheduled_tokens={'a': 4, 'b': 128},
            total_num_scheduled_tokens=132, scheduled_spec_decode_tokens={})
        self.assertEqual(runner._prepare_inputs(schedule, 2, 0), (2, 0, False))
        self.assertEqual(runner._scheduled_token_batch.query_start_loc, (0, 4, 132))
        self.assertEqual(runner.calls, 1)
        runner._native_transfers.audit_inputs = None
        self.assertEqual(runner._prepare_inputs(schedule, 2, 0), (2, 0, False))
        self.assertIsNone(runner._scheduled_token_batch)
        self.assertEqual(runner.calls, 2)
