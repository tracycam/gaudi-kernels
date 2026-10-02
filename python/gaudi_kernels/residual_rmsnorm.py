"""Explicit BF16 residual boundary, followed by FP32 RMS statistics.

The extension must be built/loaded explicitly. No CPU fallback or implicit load.
No training, in-place aliasing, quantization, or Gemma weight offset is implied.
"""
import torch


def pure_rmsnorm_bf16(x, gamma, epsilon=1e-6):
    """Pure [M,H] RMSNorm, FP32 statistics/products, one final BF16 output.

    No residual addition or intermediate BF16 product. Positive H<=8192,
    positive signed-i32 M, contiguous BF16 inference inputs; epsilon finite >0.
    """
    return torch.ops.gaudi_kernels._pure_rmsnorm_bf16(x, gamma, float(epsilon))


def residual_rmsnorm_bf16(x, residual, gamma, epsilon=1e-6):
    """Return ``(residual_out, norm_out)`` as contiguous [M,H] matrices.

    ``residual_out = RN_BF16(x + residual)``. Statistics use this rounded value;
    the FP32 normalized value multiplies BF16 gamma before final BF16 rounding.
    H is bounded to 8192; all inputs must be contiguous BF16 inference tensors.
    This numerical contract does not promise bitwise vendor-RMSNorm equality.
    """
    if x.ndim != 2 or tuple(x.shape) != tuple(residual.shape):
        raise ValueError('x and residual require the same [M,H] matrix shape')
    return torch.ops.gaudi_kernels._residual_rmsnorm_bf16(x, residual, gamma, float(epsilon))



def residual_rmsnorm_per_row_a8(x, residual, gamma, epsilon=1e-6):
    """Return residual BF16 [M,H], native A8 [M,H], compensated FP32 [M,1].

    The BF16 norm rounding boundary is retained in local memory, without an HBM
    norm output. Logical scale is max(abs(norm),1e-10)*RN32(1/448); published
    native scale is twice this. Native bytes represent half of OCP E4M3FN RNE.
    This is explicit activation quantization and requires a compatible consumer.
    """
    return torch.ops.gaudi_kernels._residual_rmsnorm_per_row_a8(
        x, residual, gamma, float(epsilon))


def residual_rmsnorm_block128_a8(x, residual, gamma, epsilon=1e-6):
    """Explicit block128 policy: residual, A8 [G,M,128], FP32 scales [G,M,1].

    G=ceil(H/128). Each block has its own logical scale and native compensation
    as above; zero tail padding belongs to this activation layout. This is not
    the per-row A8 contract and cannot feed an ordinary separable-scale GEMM.
    """
    return torch.ops.gaudi_kernels._residual_rmsnorm_block128_a8(
        x, residual, gamma, float(epsilon))


def reshape_bf16_graph(tensor, shape):
    """Emit a logical reshape in the current graph, without a Torch lazy view.

    This functional CustomOp represents activation geometry only. It is not an
    in-place KV alias API and does not promise reuse of an arbitrary input cache.
    """
    return torch.ops.gaudi_kernels._norm_reshape(tensor, list(shape))
