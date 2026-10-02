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


def input_reason(x, residual, gamma, epsilon, *, norm_policy, activation_policy,
                 same_input_audit_active=False):
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
    if not _matrix(x, (1, 6144), torch.bfloat16):
        return 'input_m1_bf16_6144'
    if not _matrix(residual, (1, 6144), torch.bfloat16, x.device):
        return 'residual_m1_bf16_6144'
    if not _matrix(gamma, (6144,), torch.bfloat16, x.device):
        return 'gamma_bf16_6144'
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        return 'epsilon'
    if not math.isfinite(epsilon) or not 0 < epsilon <= 3.4028234663852886e38:
        return 'epsilon'
    if struct.unpack('f', struct.pack('f', epsilon))[0] <= 0:
        return 'epsilon_fp32_underflow'
    return None


def produce(x, residual, gamma, epsilon, *, norm_policy, activation_policy,
            same_input_audit_active=False):
    reason = input_reason(x, residual, gamma, epsilon, norm_policy=norm_policy,
                          activation_policy=activation_policy,
                          same_input_audit_active=same_input_audit_active)
    if reason is not None:
        raise ValueError('norm-QKV probe rejected before computation: ' + reason)
    rr, q, scales = residual_rmsnorm_block128_a8(x, residual, gamma, epsilon)
    return NormBlock128Activation(rr, q, scales)


def project(activation, prepared, bias=None, *, reduce_op=None, audit_tensors=None):
    """Consume native [48,1,128] and FP32 [48,1,1] directly, without re-quantizing.

    audit_tensors retains consumer intermediates only. It cannot satisfy the
    production FP32-v1 producer+consumer certificate without a true norm witness.
    """
    if not isinstance(activation, NormBlock128Activation):
        raise TypeError('explicit NormBlock128Activation required')
    if (prepared.n, prepared.k) != (3392, 6144):
        raise ValueError('probe requires the TP8 MiMo local QKV shape')
    device = activation.residual.device
    if (not _matrix(activation.residual, (1, 6144), torch.bfloat16)
            or not _matrix(activation.q_native, (48, 1, 128), torch.float8_e4m3fn, device)
            or not _matrix(activation.scales, (48, 1, 1), torch.float32, device)):
        raise ValueError('invalid explicit norm block128 payload')
    return linear_block_fp8_quantized(activation.q_native, activation.scales, prepared,
                                     bias, reduce_op=reduce_op, audit_tensors=audit_tensors)


def residual_norm_qkv(x, residual, gamma, epsilon, prepared, bias=None, *,
                      norm_policy, activation_policy, reduce_op=None,
                      same_input_audit_active=False, audit_tensors=None):
    """Complete probe outputs (residual BF16, QKV BF16), using existing GUIDs.

    Production-sized M>1, no residual, and every audit-enabled model call must
    remain on the separate path until a different explicit plan is qualified.
    This function raises; it never silently falls back after consuming inputs.
    """
    if (prepared.n, prepared.k) != (3392, 6144):
        raise ValueError('probe requires the TP8 MiMo local QKV shape')
    activation = produce(x, residual, gamma, epsilon, norm_policy=norm_policy,
        activation_policy=activation_policy, same_input_audit_active=same_input_audit_active)
    y = project(activation, prepared, bias, reduce_op=reduce_op, audit_tensors=audit_tensors)
    return activation.residual, y
