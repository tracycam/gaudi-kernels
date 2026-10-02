"""Experimental original-width N512-native-v2 load-time packing.

This module does not change the default N256-v1 framework operator. Native
callers opt in through csrc/ops/mxfp4_overhead.hpp. No expanded weight is kept.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedN512:
    weight: np.ndarray
    scales: np.ndarray
    n: int
    k: int
    layout_version: int = 2


def prepare(rows, scales, k=None):
    """Lossless CPU preparation, independent of M and runtime split count.

    Input rows are uint8[N, ceil(K/2)] with K-even in the low nibble;
    scales are uint8[N, ceil(K/32)] with E8M0 codes 2..252 only.
    Weight byte shape is [ceil(N/512)*ceil(K/32)*32, 256]; scale shape is
    [ceil(N/512)*ceil(K/32), 512]. Native callers describe weight with the
    Synapse packed_mxfp4 dtype and scales with uint8. These bytes must never
    be passed to N256-v1, or to an older N512 layout without its permutation.
    """
    rows, scales = np.asarray(rows), np.asarray(scales)
    weight, exponents = pack(rows, scales, k)
    n, packed_k = rows.shape
    return PreparedN512(weight, exponents, n, packed_k * 2 if k is None else k)


def pack(rows, scales, k=None):
    """Return prepared byte arrays; prefer prepare() to retain version metadata."""
    rows, scales = np.asarray(rows), np.asarray(scales)
    if rows.dtype != np.uint8 or scales.dtype != np.uint8 or rows.ndim != 2:
        raise ValueError("CPU uint8 rows/scales required")
    n, packed_k = rows.shape
    k = packed_k * 2 if k is None else k
    if (not isinstance(k, (int, np.integer)) or n < 1 or k < 1
            or packed_k != (k + 1) // 2 or scales.shape != (n, (k + 31) // 32)):
        raise ValueError("MXFP4 row/scales/logical K shape mismatch")
    if np.any((scales < 2) | (scales > 252)):
        raise ValueError("E8M0 domain is 2..252")
    if k % 2 and np.any(rows[:, -1] & 240):
        raise ValueError("unused odd-K nibble must be zero")
    blocks, prepared_k = (n + 511) // 512, (k + 31) // 32 * 32
    weight = np.zeros((blocks, prepared_k, 256), np.uint8)
    exponents = np.full((blocks, prepared_k // 32, 512), 127, np.uint8)
    lanes = np.arange(512)
    # A native BF16 MAC FP32 store enumerates even physical lanes, then odd.
    # This load-time permutation makes those stores follow logical N directly.
    permutation = (lanes // 128) * 128 + (lanes % 128) // 2 + (lanes % 2) * 64
    for block in range(blocks):
        count = min(512, n - block * 512)
        source = rows[block * 512:block * 512 + count]
        codes = np.zeros((512, packed_k * 2), np.uint8)
        codes[:count, ::2], codes[:count, 1::2] = source & 15, source >> 4
        codes = codes[permutation]
        for pair in (0, 1):
            low = codes[pair * 256:pair * 256 + 128, :k].T
            high = codes[pair * 256 + 128:pair * 256 + 256, :k].T
            weight[block, :k, pair * 128:(pair + 1) * 128] = low | (high << 4)
        padded = np.full((512, (k + 31) // 32), 127, np.uint8)
        padded[:count] = scales[block * 512:block * 512 + count]
        exponents[block, :(k + 31) // 32] = padded[permutation].T
    return weight.reshape(blocks * prepared_k, 256), exponents.reshape(-1, 512)


def unpack(weight, exponents, n, k):
    """Invert N512-native-v2 preparation for loader verification."""
    blocks, prepared_k = (n + 511) // 512, (k + 31) // 32 * 32
    weight = np.asarray(weight).reshape(blocks, prepared_k, 256)
    exponents = np.asarray(exponents).reshape(blocks, prepared_k // 32, 512)
    rows = np.zeros((n, (k + 1) // 2), np.uint8)
    scales = np.empty((n, (k + 31) // 32), np.uint8)
    lanes = np.arange(512)
    inverse = (lanes // 128) * 128 + (lanes % 64) * 2 + (lanes % 128) // 64
    for block in range(blocks):
        count = min(512, n - block * 512)
        codes = np.zeros((512, k + 1), np.uint8)
        for pair in (0, 1):
            packed = weight[block, :k, pair * 128:(pair + 1) * 128]
            codes[pair * 256:pair * 256 + 128, :k] = (packed & 15).T
            codes[pair * 256 + 128:pair * 256 + 256, :k] = (packed >> 4).T
        codes = codes[inverse]
        rows[block * 512:block * 512 + count] = codes[:count, :k:2]
        rows[block * 512:block * 512 + count, :k // 2] |= codes[:count, 1:k:2] << 4
        scales[block * 512:block * 512 + count] = exponents[block, :(k + 31) // 32].T[inverse][:count]
    return rows, scales
