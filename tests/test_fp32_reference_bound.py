import unittest
import torch

from gaudi_kernels.serving.diagnostics.fp32_reference import dot_error_bound


class ForwardBoundTests(unittest.TestCase):
    def test_cancellation_does_not_turn_relative_error_into_bug(self):
        products = torch.tensor([[1., -1., 1e-7], [2., 3., 4.]], dtype=torch.float32)
        reference = products.sum(-1)
        candidate = torch.tensor([0., 9.000001])
        self.assertTrue(bool(((candidate-reference).abs()<=dot_error_bound(products)).all()))
        self.assertFalse(bool(((candidate.flip(0)-reference).abs()<=dot_error_bound(products)).all()))

    def test_no_fp64_precision_gate(self):
        self.assertEqual(dot_error_bound(torch.ones(2,6144)).dtype,torch.float32)
        with self.assertRaises(ValueError):
            dot_error_bound(torch.ones(2,6144,dtype=torch.float64))
