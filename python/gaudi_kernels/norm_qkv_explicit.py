"""Experimental explicit norm→block128→QKV graph composition, never installed.

This is a bounded probe interface, not a production policy. No tensor identity
cache, module patch, CPU arithmetic fallback, or library load is performed.
The current fused producer has a serial 48-block M1 output loop; actual complete
block128-QKV performance and the fused producer's staged audit remain unqualified.
"""
from dataclasses import dataclass
import math
import struct
import torch
from .residual_rmsnorm import residual_rmsnorm_block128_a8
from .block_fp8 import linear_block_fp8_quantized
from . import norm_grid24_binding as _grid24_binding
from .engine.artifact_pins import PINS as _ARTIFACT_PINS

for _k in tuple(_grid24_binding.PINS):
    _grid24_binding.PINS[_k] = _ARTIFACT_PINS['.'.join(_k)]


@dataclass(frozen=True)
class NormBlock128Activation:
    """Caller-owned tensors passed explicitly to one compatible consumer.

    The BF16 norm rounding occurs in the producer's VLM before quantization;
    it is not materialized here. This payload is not an audit certificate.
    """
    residual: torch.Tensor
    q_native: torch.Tensor
    scales: torch.Tensor


def _matrix(tensor, shape, dtype, device=None):
    return (isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape
            and tensor.dtype == dtype and tensor.is_contiguous()
            and not tensor.requires_grad and tensor.device.type in ('hpu', 'meta')
            and (device is None or tensor.device == device))


def _valid_rows(m, max_rows=16):
    return isinstance(m, int) and isinstance(max_rows, int) and 1 <= m <= max_rows <= 16


def _valid_hidden(h):
    return isinstance(h, int) and 256 <= h <= 8192 and h % 256 == 0


def input_reason(x, residual, gamma, epsilon, *, norm_policy, activation_policy,
                 same_input_audit_active=False, max_rows=16):
    """Metadata-only decision made before the first candidate operation.

    A model adapter must also qualify its exact classes/dispatch/hooks and its
    sole QKV consumer before calling this probe. No partially computed fallback.
    """
    if norm_policy != 'fp32':
        return 'norm_policy'
    if activation_policy != 'per_block_fp8':
        return 'activation_policy'
    if same_input_audit_active:
        return 'fused_producer_audit_witness_unqualified'
    m = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim == 2 else 0
    h = x.shape[1] if isinstance(x, torch.Tensor) and x.ndim == 2 else 0
    if not _valid_rows(m, max_rows) or not _valid_hidden(h) or not _matrix(x, (m, h), torch.bfloat16):
        return 'input_m1_bf16_6144'
    if not _matrix(residual, (m, h), torch.bfloat16, x.device):
        return 'residual_m1_bf16_6144'
    if not _matrix(gamma, (h,), torch.bfloat16, x.device):
        return 'gamma_bf16_6144'
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        return 'epsilon'
    if not math.isfinite(epsilon) or not 0 < epsilon <= 3.4028234663852886e38:
        return 'epsilon'
    if struct.unpack('f', struct.pack('f', epsilon))[0] <= 0:
        return 'epsilon_fp32_underflow'
    return None


def produce(x, residual, gamma, epsilon, *, norm_policy, activation_policy,
            same_input_audit_active=False, max_rows=16):
    reason = input_reason(x, residual, gamma, epsilon, norm_policy=norm_policy,
                          activation_policy=activation_policy,
                          same_input_audit_active=same_input_audit_active,
                          max_rows=max_rows)
    if reason is not None:
        raise ValueError('norm-QKV probe rejected before computation: ' + reason)
    rr, q, scales = residual_rmsnorm_block128_a8(x, residual, gamma, epsilon)
    return NormBlock128Activation(rr, q, scales)


def project(activation, prepared, bias=None, *, reduce_op=None, audit_tensors=None):
    """Consume native [H//128,M,128] and FP32 [H//128,M,1] directly, without re-quantizing.

    audit_tensors retains consumer intermediates only. It cannot satisfy the
    production FP32-v1 producer+consumer certificate without a true norm witness.
    """
    if not isinstance(activation, NormBlock128Activation):
        raise TypeError('explicit NormBlock128Activation required')
    m = activation.residual.shape[0] if isinstance(activation.residual, torch.Tensor) and activation.residual.ndim == 2 else 0
    h = activation.residual.shape[1] if isinstance(activation.residual, torch.Tensor) and activation.residual.ndim == 2 else 0
    if not _valid_rows(m) or not _valid_hidden(h) or prepared.k != h or prepared.n <= 0:
        raise ValueError('probe requires matching hidden dimension and positive output width')
    device = activation.residual.device
    kb = h // 128
    if (not _matrix(activation.residual, (m, h), torch.bfloat16)
            or not _matrix(activation.q_native, (kb, m, 128), torch.float8_e4m3fn, device)
            or not _matrix(activation.scales, (kb, m, 1), torch.float32, device)):
        raise ValueError('invalid explicit norm block128 payload')
    return linear_block_fp8_quantized(activation.q_native, activation.scales, prepared,
                                     bias, reduce_op=reduce_op, audit_tensors=audit_tensors)


def residual_norm_qkv(x, residual, gamma, epsilon, prepared, bias=None, *,
                      norm_policy, activation_policy, reduce_op=None,
                      same_input_audit_active=False, audit_tensors=None, max_rows=16):
    """Complete probe outputs (residual BF16, QKV BF16), using existing GUIDs.

    Production-sized M>max_rows, no residual, and every audit-enabled model call must
    remain on the separate path until a different explicit plan is qualified.
    This function raises; it never silently falls back after consuming inputs.
    """
    h = x.shape[1] if isinstance(x, torch.Tensor) and x.ndim == 2 else 0
    if not _valid_hidden(h) or prepared.k != h or prepared.n <= 0:
        raise ValueError('probe requires matching hidden dimension and positive output width')
    activation = produce(x, residual, gamma, epsilon, norm_policy=norm_policy,
        activation_policy=activation_policy, same_input_audit_active=same_input_audit_active,
        max_rows=max_rows)
    y = project(activation, prepared, bias, reduce_op=reduce_op, audit_tensors=audit_tensors)
    return activation.residual, y
