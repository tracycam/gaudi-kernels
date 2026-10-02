import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gaudi_kernels.engine import inventory


class InventoryTests(unittest.TestCase):
    def tearDown(self):
        inventory.stop()

    def test_reader_is_recorded_but_secret_value_is_not(self):
        with tempfile.TemporaryDirectory() as folder:
            inventory.start(folder)
            try:
                with patch.dict(os.environ, {'GK_TEST_READ': 'secret-that-must-not-be-saved'}):
                    exec(compile('import os; observed = os.getenv("GK_TEST_READ")',
                         str(Path(folder) / 'producer.py'), 'exec'), {})
            finally:
                inventory.stop()
            report = inventory.snapshot(folder, mapped_paths=[])
            self.assertTrue(any(row['name'] == 'GK_TEST_READ' for row in report['environment_reads']))
            self.assertNotIn('secret-that-must-not-be-saved', str(report))
            self.assertFalse(report['tracking_active'])

    def test_exception_restores_environment_class_before_timing(self):
        original = os._Environ.__getitem__
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,
                {}):
            from types import SimpleNamespace
            stub=SimpleNamespace(startup=SimpleNamespace(inventory=True))
            self.context_patch=patch("gaudi_kernels.engine.context.context", return_value=stub)
            self.context_patch.start()
            self.addCleanup(self.context_patch.stop)
            @inventory.track_selection
            def fails():
                raise ValueError('test rejection')
            with self.assertRaises(ValueError):
                fails()
            self.assertIs(os._Environ.__getitem__, original)

    def test_declared_library_is_not_called_mapped_by_assumption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'example.so'
            path.write_bytes(b'host metadata fixture')
            with patch.dict(os.environ, {'GC_KERNEL_PATH': str(path)}):
                report = inventory.snapshot(folder, mapped_paths=[])
                self.assertFalse(report['tpc_libraries'][0]['mapped'])
                report = inventory.snapshot(folder, mapped_paths=[str(path)])
                self.assertTrue(report['tpc_libraries'][0]['mapped'])


if __name__ == '__main__':
    unittest.main()
