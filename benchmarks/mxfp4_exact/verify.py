"""Independent Fraction oracle for the native build of the TPC integer core."""
import argparse
import ctypes
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import numpy as np

Q = [Fraction(x, 2) for x in (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)]

def pow2(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)

def f32_fraction(bits):
    exponent, mantissa = (bits >> 23) & 255, bits & 0x7fffff
    assert exponent != 255
    value = Fraction(mantissa if not exponent else mantissa + 0x800000) * pow2((exponent or 1) - 150)
    return -value if bits >> 31 else value

def bf16(bits):
    return f32_fraction(bits << 16)

def round_fraction(value):
    """Nearest representable value by exact distance, independently of limb code."""
    if not value:
        return 0
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    if value >= pow2(128) - pow2(103):
        return sign | 0x7f800000
    try:
        nearby = struct.unpack('<I', struct.pack('<f', float(value)))[0]
    except OverflowError:
        nearby = 0x7f7fffff
    nearby = min(nearby, 0x7f7fffff)
    options = range(max(0, nearby - 2), min(0x7f7fffff, nearby + 2) + 1)
    result = min(options, key=lambda b: (abs(f32_fraction(b) - value), b & 1))
    return sign | result

def oracle(a, q, s, bias=None):
    total = sum((bf16(int(x)) * Q[int(c)] * pow2(int(e) - 127) for x, c, e in zip(a, q, s)), Fraction())
    if bias is not None:
        total += f32_fraction(bias)
    return round_fraction(total), total

class Native:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library.resolve()))
        for name in ('exact_dot', 'certificate', 'fp32_safe'):
            function = getattr(self.lib, name)
            function.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_uint32, ctypes.c_int]
            function.restype = ctypes.c_uint32
    def call(self, name, a, q, s, bias=None):
        a, q, s = np.asarray(a, np.uint16), np.asarray(q, np.uint8), np.asarray(s, np.uint8)
        return getattr(self.lib, name)(a.ctypes.data, q.ctypes.data, s.ctypes.data, len(a), bias or 0, bias is not None)

def f32_round(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]

def float_fast(a, q, s, bias):
    """Ordinary FP32 product/add reference, only used after exact certificate."""
    total = np.float32(0)
    for x, c, e in zip(a, q, s):
        product = float(bf16(x) * Q[c] * pow2(e - 127))
        total = np.float32(total + np.float32(product))
    if bias is not None:
        total = np.float32(total + np.float32(f32_round(bias)))
    return struct.unpack('<I', total.tobytes())[0]

def verify(native, old_case=None, output=None):
    rng = random.Random(27092026)
    counts = {'cases': 0, 'terms': 0, 'certificate_accept': 0, 'certificate_reject': 0, 'fp32_fast_safe': 0}
    named = []
    def check(a, q, s, bias=None, name=None):
        expected, rational = oracle(a, q, s, bias)
        actual = native.call('exact_dot', a, q, s, bias)
        assert actual == expected, (name, hex(actual), hex(expected), a, q, s, bias)
        accepted = native.call('certificate', a, q, s, bias)
        if accepted:
            assert float_fast(a, q, s, bias) == expected, ('unsafe certificate', name)
        counts['certificate_accept' if accepted else 'certificate_reject'] += 1
        counts['fp32_fast_safe'] += bool(native.call('fp32_safe', a, q, s, bias))
        counts['cases'] += 1
        counts['terms'] += len(a)
        if name:
            named.append({'case': name, 'expected_bits': f'{expected:08x}', 'actual_bits': f'{actual:08x}', 'certificate': bool(accepted),
                          'exact_numerator': str(rational.numerator), 'exact_denominator': str(rational.denominator)})
        return actual
    # BF16 subnormal contributions sum to FP32 normal, or survive cancellation.
    check([1] * 128, [2] * 128, [127] * 128, name='128_subnormal_products_make_min_normal')
    check([0x7f7f, 1, 0x7f7f], [7, 2, 15], [252, 252, 252], name='beyond_fp32_products_cancel_keep_small_normal')
    check([0x7e80, 0x3f80, 0x7e80], [2, 2, 10], [252, 127, 252], name='fp64_loses_one_after_2p251_cancellation')
    check([1], [1], [111], name='positive_half_min_subnormal_ties_to_zero')
    check([1, 1], [1, 1], [111, 2], name='half_min_subnormal_plus_2pm259_rounds_up')
    check([0x8001], [1], [111], name='negative_half_min_subnormal_ties_to_negative_zero')
    check([1], [1], [111], bias=0x007fffff, name='max_subnormal_half_ulp_to_min_normal')
    check([0x3f80], [2], [103], bias=0x3f800000, name='one_plus_half_ulp_even')
    check([0x3f80, 1], [2, 1], [103, 2], bias=0x3f800000, name='one_plus_half_ulp_plus_tiny')
    check([0x3f80], [2], [103], bias=0x3f800001, name='odd_significand_half_ulp_rounds_even')
    check([0x7300], [2], [127], bias=0x7f7fffff, name='max_finite_plus_half_ulp_overflows')
    check([0x7300, 1], [2, 9], [127, 2], bias=0x7f7fffff, name='just_below_overflow_tie_stays_finite')
    check([0x3f80, 0xbf80], [2, 2], [127, 127], name='exact_zero')
    check([0] * 64, [7] * 64, [252] * 64, bias=0x80000000, name='zero_and_negative_zero_bias')
    # Every signed BF16 subnormal mantissa, all E2M1 codes, representative extremes.
    for mantissa in range(1, 128):
        for sign in (0, 0x8000):
            for scale in (0, 1, 2, 9, 10, 110, 111, 127, 252, 253, 254):
                for code in range(16):
                    check([sign | mantissa], [code], [scale])
    for scale in range(255):
        for code in range(16):
            for activation in (0, 1, 127, 128, 129, 0x7f7f, 0x8001, 0xff7f):
                check([activation], [code], [scale])
    for _ in range(1200):
        k = rng.choice([1, 2, 3, 31, 32, 33, 127, 257])
        a = [rng.randrange(0x7f80) | rng.choice([0, 0x8000]) for _ in range(k)]
        q = [rng.randrange(16) for _ in range(k)]
        s = [rng.randrange(0, 255) for _ in range(k)]
        bias = rng.randrange(0x7f800000) | rng.choice([0, 0x80000000]) if rng.randrange(2) else None
        check(a, q, s, bias)
    # Cancellation pairs with an exact residual below FP64 precision.
    for _ in range(500):
        a0 = rng.randrange(1, 0x7f80)
        q0 = rng.randrange(1, 8)
        s0 = rng.randrange(0, 255)
        check([a0, 1, a0], [q0, rng.randrange(1, 8), q0 | 8], [s0, rng.randrange(0, 255), s0])
    # Ordinary exactness-certified examples, including arbitrary reduction order.
    for _ in range(500):
        k = rng.randrange(1, 129)
        a = [rng.choice([0x3f00, 0x3f80, 0xbf00, 0xbf80]) for _ in range(k)]
        q, s = [rng.randrange(16) for _ in range(k)], [127] * k
        check(a, q, s)
        for _ in range(3):
            order = list(range(k)); rng.shuffle(order)
            assert native.call('exact_dot', [a[i] for i in order], [q[i] for i in order], [s[i] for i in order]) == native.call('exact_dot', a, q, s)
    # Exercise the proved INT_MAX count without an impractical billion-step loop.
    repeated = native.lib.repeat_cancel
    repeated.argtypes = [ctypes.c_uint16, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint32]
    repeated.restype = ctypes.c_uint32
    for scale in (0, 1, 127, 253, 254):
        for bias in (0, 1, 0x80000001, 0x00800000, 0x3f800000, 0x7f7fffff):
            assert repeated(0x7f7f, 7, scale, 2147483647, bias) == bias
    recovered = None
    if old_case:
        n, k, m, kp = 513, 257, 3, 288
        x = np.fromfile(old_case / 'e0_activation.bin', np.uint16).reshape(m, k)
        w = np.fromfile(old_case / 'e0_packed.bin', np.uint8).reshape(2, kp, 256)
        e = np.fromfile(old_case / 'e0_e8m0.bin', np.uint8).reshape(2, kp // 32, 512)
        old = np.fromfile(old_case / 'e0_output.bin', np.uint32).reshape(m, n)
        restored = np.empty((m, n), np.uint32)
        for col in range(n):
            physical = (col // 128) * 128 + (col % 64) * 2 + (col % 128) // 64
            q = [(int(w[physical // 512, z, (physical % 512) // 256 * 128 + physical % 128]) >> (4 * ((physical % 256) // 128))) & 15 for z in range(k)]
            s = [int(e[physical // 512, z // 32, physical % 512]) for z in range(k)]
            for row in range(m):
                restored[row, col] = check(x[row].tolist(), q, s)
        recovered = {'outputs': m * n, 'original_device_bits_differ': int(np.count_nonzero(restored != old)), 'new_integer_cpu_mismatches': 0,
                     'source_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in old_case.glob('*.bin')}}
        if output:
            restored.tofile(output / 'old_mixed_fixture_exact_cpu_u32.bin')
    return {'status': 'CPU_verified_only_no_device_validation', 'all_cpu_checks_pass': True, **counts, 'named': named, 'old_fixture': recovered, 'int_max_repeat_cancellation_cases': 30}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--old-case', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args();a.output.mkdir(parents=True, exist_ok=True)
    result = verify(Native(a.library), a.old_case, a.output)
    (a.output / 'cpu-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('named', 'old_fixture')}))
