# SPDX-License-Identifier: Apache-2.0
"""Multi-rank ownership of validation files and restoration on disable."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.validation.executor.api_worker_extension import ApiWorkerExtension


class LogitCaptureTest(unittest.TestCase):
    def test_shared_run_directory_has_independent_rank_ownership(self):
        with TemporaryDirectory() as temporary:
            context = SimpleNamespace(startup=SimpleNamespace(run_dir=Path(temporary)))
            workers = []
            with patch('gaudi_kernels.engine.context.context', return_value=context):
                for rank in range(8):
                    worker = ApiWorkerExtension()
                    original = lambda *args: None
                    worker.rank = rank
                    worker.model_runner = SimpleNamespace(_execute_model_generic=original)
                    worker.framework_target_logits('same-teacher-input')
                    workers.append((worker, original))
            self.assertEqual(len(list(Path(temporary).glob('target-logits/same-teacher-input/rank*'))), 8)
            for worker, original in workers:
                disabled = worker.framework_target_logits()
                self.assertIs(worker.model_runner._execute_model_generic, original)
                self.assertEqual(disabled['frames'], [])

    def test_invalid_capture_is_rejected_before_accessing_runner(self):
        worker = ApiWorkerExtension()
        for tag, calls in (('../escape', 1), ('valid', 0), ('valid', 129)):
            with self.assertRaises(ValueError):
                worker.framework_target_logits(tag, calls)


if __name__ == '__main__':
    unittest.main()
