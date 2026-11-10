"""Load-time conversion from OCP E4M3FN to Gaudi2 E4M3 half range.

The returned bytes are Gaudi2 IEEE-specials bit patterns, not an OCP tensor
despite using PyTorch's one-byte E4M3FN storage type.  Halving values near
zero requires rounding: odd low-end significands cannot be represented exactly
after the scale is doubled.  The caller must retain the checkpoint as its
independent numerical reference.
"""

import numpy as np
import torch


def reencode_e4m3fn_half(weight, scales):
    """Return (encoded bytes, doubled FP32 scales, rounded element count).

    Inputs must be contiguous CPU tensors.  Neither input is modified, and
    nonfinite checkpoint codes or overflowing/nonfinite scales are rejected.
    The shape of each tensor is preserved; layout packing is a separate step.
    """
    if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
            or weight.dtype != torch.float8_e4m3fn or not weight.is_contiguous()
            or not weight.numel() or weight.requires_grad):
        raise ValueError("expected a nonempty contiguous CPU E4M3FN weight")
    if (not isinstance(scales, torch.Tensor) or scales.device.type != "cpu"
            or scales.dtype != torch.float32 or not scales.is_contiguous()
            or not scales.numel() or scales.requires_grad
            or not bool(torch.isfinite(scales).all()) or bool((scales < 0).any())):
        raise ValueError("expected nonempty contiguous finite nonnegative CPU FP32 scales")
    adjusted_scales = scales * 2.0
    if not bool(torch.isfinite(adjusted_scales).all()):
        raise ValueError("doubling FP8 scales overflows FP32")

    # Exact halves are representable except for odd low-end significands;
    # those use round-to-nearest, ties-to-even in the Gaudi2 encoding.
    codes = np.arange(256, dtype=np.uint16)
    magnitude = codes & 0x7f
    half = np.where(magnitude >= 16, magnitude - 8,
                    (magnitude >> 1) + ((magnitude & 1) & ((magnitude >> 1) & 1)))
    table = ((codes & 0x80) | half).astype(np.uint8)
    source = weight.detach().view(torch.uint8).numpy().reshape(-1)
    result = np.empty(source.shape, dtype=np.uint8)
    rounded = 0
    # Bound temporary memory even for large checkpoint matrices.
    for start in range(0, source.size, 8 * 1024 * 1024):
        raw = source[start:start + 8 * 1024 * 1024]
        mag = raw & 0x7f
        if np.any(mag == 0x7f):
            raise ValueError("nonfinite OCP E4M3FN checkpoint code")
        rounded += int(np.count_nonzero((mag < 16) & ((mag & 1) != 0)))
        np.take(table, raw, out=result[start:start + raw.size])
    encoded = torch.from_numpy(result.reshape(tuple(weight.shape))).view(torch.float8_e4m3fn)
    return encoded, adjusted_scales.contiguous(), rounded
