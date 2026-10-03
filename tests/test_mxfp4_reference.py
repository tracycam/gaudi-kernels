import unittest
import torch
from gaudi_kernels.mxfp4_reference import (
    MXFP4Weights, linear_reference, moe_reference, quantize_mxfp8)


class RaggedMXFP4ReferenceTests(unittest.TestCase):
    def weights(self, e, n, k):
        return MXFP4Weights(torch.randint(0, 256, (e, n, (k+1)//2), dtype=torch.uint8),
                            torch.randint(122, 126, (e, n, (k+31)//32), dtype=torch.uint8), k)

    def test_block_scales_cannot_be_one_epilogue_scale(self):
        w = MXFP4Weights(torch.full((1, 1, 32), 0x22, dtype=torch.uint8),
                        torch.tensor([[[127, 128]]], dtype=torch.uint8), 64)
        x = torch.ones((1, 64), dtype=torch.bfloat16)
        x[:, 32:] = -1
        # Unscaled dot is zero, but the correct scaled sum is -32.
        for policy in ('w4a16', 'w4a8_mxfp8'):
            self.assertEqual(linear_reference(x, w, 0, policy=policy).item(), -32.)

    def test_tail_quantization_and_scale_are_per_block(self):
        x = torch.ones((2, 35), dtype=torch.bfloat16)
        x[:, 32:] = 128
        q, s = quantize_mxfp8(x)
        self.assertEqual(tuple(q.shape), (2, 35))
        self.assertEqual(tuple(s.shape), (2, 2))
        self.assertTrue((s[:, 1].int()-s[:, 0].int() == 7).all())
        restored = q.float() * torch.exp2(s.float()-127).repeat_interleave(32, -1)[:, :35]
        torch.testing.assert_close(restored, x.float(), rtol=0, atol=0)

    def test_ragged_expert_counts_and_token_permutation(self):
        torch.manual_seed(321)
        # Counts [4,3,2,1,0]: no per-expert padding, one inactive expert.
        x = torch.randn(4, 35).to(torch.bfloat16)
        ids = torch.tensor([[0,1,2,3], [0,1,2,-1], [0,1,-1,-1], [0,-1,-1,-1]])
        routing = (ids >= 0).float()/4
        gp, down = self.weights(5, 18, 35), self.weights(5, 35, 9)
        order = torch.tensor([3,1,0,2])
        for policy in ('w4a16', 'w4a8_mxfp8'):
            y = moe_reference(x, ids, routing, gp, down, policy=policy)
            yp = moe_reference(x[order], ids[order], routing[order], gp, down, policy=policy)
            torch.testing.assert_close(yp, y[order], rtol=2e-5, atol=2e-5)
            for row in range(4):
                single = moe_reference(x[row:row+1], ids[row:row+1], routing[row:row+1], gp, down, policy=policy)
                torch.testing.assert_close(single, y[row:row+1], rtol=2e-5, atol=2e-5)

    def test_reject_bad_routes_and_nonfinite_scale(self):
        gp, down = self.weights(2, 8, 4), self.weights(2, 4, 4)
        x = torch.ones((1,4), dtype=torch.bfloat16)
        for ids in (torch.tensor([[0,0]]), torch.tensor([[0,2]]), torch.tensor([[0,-1]])):
            with self.assertRaises(ValueError):
                moe_reference(x, ids, torch.ones((1,2)), gp, down, policy='w4a16')
        gp.scales[0,0,0] = 255
        with self.assertRaises(ValueError):
            gp.validate()

    def test_empty_and_all_masked_routes(self):
        gp, down = self.weights(1, 8, 4), self.weights(1, 4, 4)
        for t in (0, 3):
            for policy in ('w4a16', 'w4a8_mxfp8'):
                y = moe_reference(torch.ones((t,4), dtype=torch.bfloat16),
                    torch.full((t,1), -1), torch.zeros((t,1)), gp, down, policy=policy)
                torch.testing.assert_close(y, torch.zeros((t,4)), rtol=0, atol=0)
