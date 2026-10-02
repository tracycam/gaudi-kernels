"""Offline AG consumer experiment; never imported or installed by production.

The caller owns the graph, communicator and rank-major gathered tensor. This
module cannot establish collective completion or HCCL reduction equivalence.
"""
import math
import torch


def validate(gathered, residual, gamma, epsilon, policy):
    if policy not in ('vendor_boundaries', 'fp32_statistics'):
        raise ValueError('explicit norm arithmetic policy required')
    if (residual.ndim != 2 or not 0 < residual.shape[0] <= (2**31-1)//8
            or not 0 < residual.shape[1] <= 8192):
        raise ValueError('residual must be [M,H], 0<H<=8192')
    m, h = residual.shape
    if gathered.shape != (8*m, h) or gamma.shape != (h,):
        raise ValueError('rank-major gathered [8*M,H] and gamma [H] required')
    if (gathered.dtype != torch.float32 or residual.dtype != torch.bfloat16
            or gamma.dtype != torch.bfloat16):
        raise ValueError('FP32 gathered and BF16 residual/gamma required')
    if any(t.device != residual.device or not t.is_contiguous() or t.requires_grad
           for t in (gathered, residual, gamma)):
        raise ValueError('contiguous inference tensors on one device required')
    if not math.isfinite(epsilon) or not 2**-149 <= epsilon <= 3.4028234663852886e38:
        raise ValueError('positive finite FP32 epsilon required')


def consume_gathered(gathered, residual, gamma, epsilon=1e-6, *, policy):
    """Return (residual_bf16, norm_bf16) without a separate sum output/node.

    Sequential rank0..7 FP32 additions from +0 -> RN_BF16 -> BF16 residual
    addition -> chosen RMS boundaries. ``vendor_boundaries`` preserves observed
    BF16 square and numerator boundaries, not the unextracted vendor ELF's exact
    reduction/rsqrt schedule. Neither route is device-qualified yet.
    """
    validate(gathered, residual, gamma, epsilon, policy)
    if gathered.device.type != 'hpu':
        raise ValueError('HPU consumer required; CPU oracle is a separate fixture')
    return getattr(torch.ops.gaudi_ag_norm_experiment, policy)(
        gathered, residual, gamma, float(epsilon))


def gather_then_consume(local_fp32, gathered_owner, residual, gamma, gather,
                        epsilon=1e-6, *, policy):
    """Minimal injected-collective fixture; no install, mark_step, sync or view.

    ``gather(output, input)`` must record the real data dependency in the same
    captured execution domain (e.g. all_gather_into_tensor with a fixed group).
    Owners are caller-held tensors, allocated before capture. The callback must
    guarantee collective completion/dependency before the consumer may run.
    This API does not borrow tensor pointers or fabricate a local completion.
    """
    validate(gathered_owner, residual, gamma, epsilon, policy)
    if (local_fp32.shape != residual.shape or local_fp32.dtype != torch.float32
            or local_fp32.device != residual.device or not local_fp32.is_contiguous()
            or local_fp32.requires_grad):
        raise ValueError('local unreduced FP32 [M,H] producer required')
    gather(gathered_owner, local_fp32)
    return consume_gathered(gathered_owner, residual, gamma, epsilon, policy=policy)
