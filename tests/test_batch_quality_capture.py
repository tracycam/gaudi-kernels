from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from gaudi_kernels.serving.executor import production_quality


class BatchCaptureTests(unittest.TestCase):
    def test_shared_run_directory_has_distinct_rank_traces_and_preserves_results_on_drain(self):
        package=ModuleType('habana_frameworks');hpt=ModuleType('habana_frameworks.torch')
        core=ModuleType('habana_frameworks.torch.core');core.mark_step=lambda:None
        package.torch=hpt;hpt.core=core
        modules={'habana_frameworks':package,'habana_frameworks.torch':hpt,'habana_frameworks.torch.core':core}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run';root.mkdir()
            ctx=SimpleNamespace(native=SimpleNamespace(active=False),
                startup=SimpleNamespace(run_dir=root,same_input_audit=False,layer_diagnostics=False),has=lambda _:False)
            workers=[SimpleNamespace(rank=r,model_runner=SimpleNamespace(
                _native_transfers=SimpleNamespace(audit_inputs=None))) for r in range(8)]
            with patch.dict(sys.modules,modules), patch.object(torch,'hpu',SimpleNamespace(synchronize=lambda:None),create=True), \
                 patch.object(production_quality,'execution_context',lambda:ctx):
                for w in workers:
                    production_quality.configure_capture(w,'same-tag','batch',3)
                paths=[w.model_runner._production_quality_trace for w in workers]
                self.assertEqual(len(set(paths)),8)
                self.assertTrue(all(p.is_dir() for p in paths))
                workers[0].model_runner._production_quality_frames=[{'path':'frame.pt'}]
                record=production_quality.configure_capture(workers[0],None)
                self.assertEqual(record['captured_forwards'],[{'path':'frame.pt'}])
                self.assertIsNone(workers[0].model_runner._production_quality_trace)
                with self.assertRaises(FileExistsError):
                    production_quality.configure_capture(workers[0],'same-tag','batch',3)
