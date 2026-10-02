"""Explicit experimental bindings; no import-time build, device use or backend selection.

Call load_extension once before constructing/capturing the model. Set GC_KERNEL_PATH
to the BF16 and FP8 TPC libraries before importing Habana. FP8 native prepared bytes
are NOT unmodified OCP E4M3FN values. Never use this interface for block-scaled weights.
"""
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedSeparableFP8:
    """One-byte native weights; ordinary per-tensor/channel scales only."""
    weight: object
    scale: object
    bias: object
    format_version: int = 1

    def to(self, device):
        """Load-time transfer only; keep this owner alive throughout graph replay."""
        if self.format_version!=1:raise ValueError('unsupported prepared FP8 version')
        return PreparedSeparableFP8(self.weight.to(device), self.scale.to(device), self.bias.to(device), self.format_version)


def prepare_separable_fp8(weight, scale, bias=None):
    """CPU load-time OCP E4M3FN preparation; preserves one byte per weight.

    Only channels containing |q|>=256 are halved, with doubled FP32 scale.
    Tiny subnormals in those channels may round; full MAC error gates are required.
    No checkpoint block-scale conversion, activation quantization or HBM expansion.
    """
    import torch
    if weight.device.type!='cpu' or scale.device.type!='cpu':
        raise ValueError('preparation requires CPU loading tensors; no implicit device readback')
    if weight.dtype!=torch.float8_e4m3fn or weight.ndim!=2 or not weight.is_contiguous() or min(weight.shape)<=0:
        raise ValueError('expected nonempty contiguous OCP E4M3FN [N,K]')
    if scale.dtype!=torch.float32 or scale.ndim>1 or scale.numel() not in (1,weight.shape[0]):
        raise ValueError('only FP32 per-tensor or per-channel scales, never block scales')
    if not bool(torch.isfinite(scale).all()) or bool((scale<0).any()):
        raise ValueError('scales must be finite and nonnegative')
    codes=weight.view(torch.uint8);magnitude=codes&127
    if bool((magnitude==127).any()):raise ValueError('nonfinite E4M3FN weight')
    adapt=(magnitude>=120).any(-1)
    a=magnitude.to(torch.int16)
    half=torch.where(a>=16,a-8,(a>>1)+((a&1)&((a>>1)&1))).to(torch.uint8)
    prepared=torch.where(adapt[:,None],(codes&128)|half,codes).contiguous().view(torch.float8_e4m3fn)
    scales=scale.reshape(-1).expand(weight.shape[0]).clone()
    scales[adapt]*=2
    if not bool(torch.isfinite(scales).all()):raise ValueError('adapted scale overflows FP32')
    if bias is None:bias=torch.zeros(weight.shape[0],dtype=torch.float32)
    if bias.device.type!='cpu' or bias.dtype!=torch.float32 or tuple(bias.shape)!=(weight.shape[0],) or not bool(torch.isfinite(bias).all()):
        raise ValueError('bias must be finite CPU FP32 [N]')
    return PreparedSeparableFP8(prepared,scales,bias.contiguous().clone())


def load_extension(path):
    import torch
    torch.ops.load_library(str(Path(path).resolve(strict=True)))


def linear_bf16(x, weight, bias=None):
    """BF16 [M,K]@[N,K], FP32 accumulation and FP32 bias before BF16 output."""
    import torch
    if bias is None:
        return torch.ops.gaudi_kernels._bf16_mm(x, weight)
    partial = torch.ops.gaudi_kernels._bf16_mm_f32(x, weight)
    return torch.ops.gaudi_kernels._bf16_bias(partial, bias)


def linear_fp8_w8a8_prepared(x, native_weight, prepared_channel_scale, bias, *, optimized=False, quantization=None):
    """Per-token dynamic A8 + native FP8 MME + FP32 scale/bias -> BF16.

    Weight must come from prepare_fp8_linear with channel range adaptation applied
    once and compensated FP32 separable channel scales. Bias is a preallocated
    FP32 [N] tensor (all zeros if absent). BF16 input is quantized: this is W8A8.
    """
    import torch
    quant = torch.ops.gaudi_kernels._fp8_quant_fast if optimized else torch.ops.gaudi_kernels._fp8_quant
    epilogue = torch.ops.gaudi_kernels._fp8_epilogue_rows if optimized else torch.ops.gaudi_kernels._fp8_epilogue
    if quantization is None:
        quantized, activation_scale = quant(x)
    else:
        from .fp8_quantizer import PreparedFP8Quantizer
        if not isinstance(quantization,PreparedFP8Quantizer):raise ValueError('prepared FP8 quantizer required')
        quantization.validate(x.device)
        operation=torch.ops.gaudi_kernels._fp8_quant_lut4 if quantization.variant=='lut4' else torch.ops.gaudi_kernels._fp8_quant_lut1
        quantized,activation_scale=operation(x,quantization.table)
    partial = torch.ops.gaudi_kernels._fp8_mm_f32(quantized, native_weight)
    return epilogue(partial, activation_scale, prepared_channel_scale, bias)


def linear_fp8(x, prepared, *, activation, optimized=False, quantization=None):
    """Explicit arithmetic contract. Use within the model's current lazy/HPU graph.

    W8A16 SRAM placement is qualified only for documented shapes. Do not keep the
    internal decoded tensor or split this chain using mark_step/synchronize.
    """
    import torch
    if not isinstance(prepared,PreparedSeparableFP8) or prepared.format_version!=1:
        raise ValueError('a version1 PreparedSeparableFP8 is required')
    if activation=='per_token_fp8':
        return linear_fp8_w8a8_prepared(x,prepared.weight,prepared.scale,prepared.bias,optimized=optimized,quantization=quantization)
    if activation=='bf16':
        if quantization is not None:raise ValueError('BF16 activation policy cannot accept an FP8 quantizer')
        decoded=torch.ops.gaudi_kernels._fp8_decode(prepared.weight)
        partial=torch.ops.gaudi_kernels._bf16_mm_f32(x,decoded)
        epilogue = torch.ops.gaudi_kernels._fp8_epilogue16_rows if optimized else torch.ops.gaudi_kernels._fp8_epilogue16
        return epilogue(partial,prepared.scale,prepared.bias)
    raise ValueError('activation must explicitly be bf16 or per_token_fp8')
