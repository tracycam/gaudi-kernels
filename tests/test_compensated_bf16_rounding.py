"""Check compensated rounding against exact BF16 midpoint decisions."""
import unittest

import numpy as np


def compensated(sum_, correction, bias):
    first = np.float32(sum_ + correction)
    first_error = np.where(
        np.abs(sum_) >= np.abs(correction),
        np.float32(np.float32(sum_ - first) + correction),
        np.float32(np.float32(correction - first) + sum_)).astype(np.float32)
    total = np.float32(first + bias)
    bias_error = np.where(
        np.abs(first) >= np.abs(bias),
        np.float32(np.float32(first - total) + bias),
        np.float32(np.float32(bias - total) + first)).astype(np.float32)
    error = np.float32(first_error + bias_error)
    bits = total.view(np.uint32)
    positive = np.where((bits & np.uint32(0x80000000)) == 0, bits + 1, bits - 1).astype(np.uint32)
    negative = np.where((bits & np.uint32(0x80000000)) == 0, bits - 1, bits + 1).astype(np.uint32)
    adjusted = np.where(error > 0, positive, np.where(error < 0, negative, bits)).astype(np.uint32)
    return np.where((bits & np.uint32(0xffff)) == 0x8000, adjusted, bits).astype(np.uint32)


def round_float32_bits_to_bf16(bits):
    top = bits >> np.uint32(16)
    low = bits & np.uint32(0xffff)
    increment = (low > 0x8000) | ((low == 0x8000) & ((top & 1) != 0))
    return (top + increment.astype(np.uint32)).astype(np.uint16)


class CompensatedRoundingTest(unittest.TestCase):
    def test_midpoints_with_residuals_in_both_directions(self):
        # Include both midpoint parities, both signs, and many exponent bands.
        lower_bits = np.arange(0x0200, 0x7f7f, 17, dtype=np.uint32)
        lower = (lower_bits << np.uint32(16)).view(np.float32)
        upper = ((lower_bits + 1) << np.uint32(16)).view(np.float32)
        midpoint = np.float32((lower.astype(np.float64) + upper.astype(np.float64)) * 0.5)
        ulp = np.nextafter(midpoint, np.float32(np.inf)) - midpoint
        for sign in (1, -1):
            for direction in (-1, 0, 1):
                for use_bias in (False, True):
                    target = np.float32(sign * midpoint)
                    bias = np.float32(target * np.float32(0.125)) if use_bias else np.zeros_like(target)
                    sum_ = np.float32(target - bias)
                    correction = np.float32(sign * direction * ulp * np.float32(0.25))
                    actual = round_float32_bits_to_bf16(compensated(sum_, correction, bias))
                    expected_mag = lower_bits + ((direction > 0) |
                        ((direction == 0) & ((lower_bits & 1) != 0))).astype(np.uint32)
                    expected = (expected_mag | (np.uint32(0x8000) if sign < 0 else 0)).astype(np.uint16)
                    np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
