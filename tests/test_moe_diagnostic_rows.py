import unittest

from gaudi_kernels.engine.config import ConfigError
from gaudi_kernels.serving.executor import moe_dispatch_runtime as dispatch


class MoEDiagnosticRowsTests(unittest.TestCase):
    def test_scope_restores_dispatch_on_exception_without_admitting_serving(self):
        dispatch.set_rows(tuple(range(1,9)))
        self.assertEqual(dispatch.select_mode('auto',12),'broadcast')
        with self.assertRaisesRegex(RuntimeError,'probe failed'):
            with dispatch.diagnostic_compact_rows((12,)):
                self.assertEqual(dispatch.select_mode('auto',12),'compact')
                self.assertTrue(dispatch.scale_tail_row_allowed(12))
                self.assertFalse(dispatch.scale_tail_row_allowed(24))
                with self.assertRaises(ConfigError):
                    dispatch.validate_rows((12,))
                raise RuntimeError('probe failed')
        self.assertEqual(dispatch.select_mode('auto',12),'broadcast')
        self.assertFalse(dispatch.scale_tail_row_allowed(12))
        self.assertEqual(dispatch.snapshot()['compact_rows'],list(range(1,9)))
        dispatch.set_rows((1,))

    def test_no_blanket_large_row_override(self):
        for rows in ((16,),(24,),(8,12,16,24),[12]):
            with self.assertRaises(ConfigError):
                with dispatch.diagnostic_compact_rows(rows):
                    pass
