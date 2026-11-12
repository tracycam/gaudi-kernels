import unittest

import torch

from gaudi_kernels.fp8_codec import reencode_e4m3fn_half


def value(magnitude):
    exponent, mantissa = divmod(magnitude, 8)
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0**(exponent - 7)


class FP8CodecTests(unittest.TestCase):
    def test_all_finite_codes_against_independent_nearest_even(self):
        # IEEE-specials E4M3 has finite magnitudes 0..0x77; the OCP
        # checkpoint additionally uses 0x78..0x7e for finite values.
        source = [code for code in range(256) if code & 0x7f != 0x7f]
        weight = torch.tensor(source, dtype=torch.uint8).view(torch.float8_e4m3fn)
        scales = torch.tensor([0.0, 0.25, 1.0e30], dtype=torch.float32)
        original = weight.view(torch.uint8).clone()
        encoded, adjusted, rounded = reencode_e4m3fn_half(weight, scales)
        expected = []
        errors = []
        for code in source:
            wanted = value(code & 0x7f) / 2.0
            distance = [abs(value(candidate) - wanted) for candidate in range(0x78)]
            best = min(distance)
            ties = [candidate for candidate, error in enumerate(distance) if error == best]
            selected = next(candidate for candidate in ties if candidate % 2 == 0) if len(ties) > 1 else ties[0]
            expected.append((code & 0x80) | selected)
            errors.append(best)
        self.assertEqual(encoded.view(torch.uint8).tolist(), expected)
        self.assertTrue(torch.equal(weight.view(torch.uint8), original))
        self.assertTrue(torch.equal(adjusted, scales * 2))
        self.assertEqual(rounded, sum(error != 0 for error in errors))
        self.assertEqual(rounded, 16)
        # The smallest nonzero OCP magnitude halves below the hardware's
        # smallest subnormal and therefore cannot be represented losslessly.
        self.assertEqual(expected[source.index(0x01)], 0x00)
        self.assertEqual(expected[source.index(0x81)], 0x80)

    def test_nonfinite_codes_and_bad_scales_fail(self):
        scales = torch.ones(1, dtype=torch.float32)
        for code in (0x7f, 0xff):
            weight = torch.tensor([0, code], dtype=torch.uint8).view(torch.float8_e4m3fn)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                reencode_e4m3fn_half(weight, scales)
            self.assertEqual(weight.view(torch.uint8).tolist(), [0, code])
        weight = torch.zeros(2, dtype=torch.uint8).view(torch.float8_e4m3fn)
        for bad in (float("nan"), float("inf"), -1.0, torch.finfo(torch.float32).max):
            with self.assertRaises(ValueError):
                reencode_e4m3fn_half(weight, torch.tensor([bad], dtype=torch.float32))

    def test_shape_dtype_and_contiguity(self):
        weight = torch.tensor([[0x08, 0x10, 0x78], [0x03, 0x7e, 0x80]],
                              dtype=torch.uint8).view(torch.float8_e4m3fn)
        scales = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
        encoded, adjusted, rounded = reencode_e4m3fn_half(weight, scales)
        self.assertEqual(encoded.shape, weight.shape)
        self.assertEqual(adjusted.shape, scales.shape)
        self.assertEqual(rounded, 1)
        self.assertEqual(encoded.view(torch.uint8).tolist(), [[4, 8, 112], [2, 118, 128]])
        with self.assertRaises(ValueError):
            reencode_e4m3fn_half(weight.t(), scales)
        with self.assertRaises(ValueError):
            reencode_e4m3fn_half(weight, scales.to(torch.float64))


if __name__ == "__main__":
    unittest.main()
