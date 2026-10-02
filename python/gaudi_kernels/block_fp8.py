"""Block128 FP8 Linear in the framework's current graph.

Loading is CPU-only and explicit. Prepared storage remains one byte per padded
weight, with the original 128x128 FP32 scale grid. Native half-range preparation
rounds tiny subnormals; it is not lossless and does not change block scales into
channel scales. No decoded weight is returned or retained.
"""
from dataclasses import dataclass, replace
import hashlib


def _hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class PreparedBlockFP8:
    weight: object
    scales: object
    bias: object
    n: int
    k: int
    checkpoint_weight_sha256: str
    checkpoint_scales_sha256: str
    rounded_half_values: int
    encoding: str = 'native_half_rne_v1'
    layout_version: int = 1
    bf16_fp32_scale_safe: bool = False
    bf16_bf16_scale_safe: bool = False

    def validate(self):
        import torch
        g = (self.k + 127) // 128
        if (self.layout_version != 1 or self.encoding != 'native_half_rne_v1'
                or not 0 < self.k <= 2**31 - 128 or not 0 < self.n <= 2**31 - 128
                or self.weight.dtype != torch.float8_e4m3fn
                or tuple(self.weight.shape) != (g, self.n, 128)
                or self.scales.dtype != torch.float32
                or tuple(self.scales.shape) != ((self.n + 127)//128, g)
                or self.bias.dtype != torch.float32 or tuple(self.bias.shape) != (self.n,)
                or len({t.device for t in (self.weight, self.scales, self.bias)}) != 1
                or any(not t.is_contiguous() or t.requires_grad for t in (self.weight, self.scales, self.bias))):
            raise ValueError('invalid native-half block128 prepared storage')

    def to(self, device):
        """Load-time transfer. Keep this owner alive throughout graph replay."""
        self.validate()
        return replace(self, weight=self.weight.to(device), scales=self.scales.to(device), bias=self.bias.to(device))


def prepare_block_fp8(weight, scales, bias=None):
    """Prepare one already-TP-sharded OCP matrix, never an adapted loader tensor.

    The caller must supply its local scale grid, including independent QKV rank
    group boundaries. Global MiMo [27136,6144]/[216,48] is deliberately rejected:
    first select one [3392,6144]/[27,48] TP group using its recorded grouping.
    K shards must start at a source block boundary. Checkpoint tensors are not
    mutated. An existing Gaudi half-range loader must be bypassed, not run twice.
    """
    import torch
    if (weight.device.type != 'cpu' or scales.device.type != 'cpu'
            or weight.dtype != torch.float8_e4m3fn or weight.ndim != 2
            or not weight.is_contiguous() or weight.requires_grad):
        raise ValueError('raw contiguous CPU OCP E4M3FN [N,K] checkpoint shard required')
    n, k = weight.shape
    if not 0 < n <= 2**31-128 or not 0 < k <= 2**31-128:
        raise ValueError('unsupported empty/oversize block FP8 shape')
    g = (k+127)//128
    if (scales.dtype != torch.float32 or tuple(scales.shape) != ((n+127)//128, g)
            or not bool(torch.isfinite(scales).all()) or bool((scales < 0).any())):
        raise ValueError('finite nonnegative FP32 local block128 scale grid required')
    raw = weight.view(torch.uint8)
    mag = (raw & 127).to(torch.int16)
    if bool((mag == 127).any()):
        raise ValueError('nonfinite OCP checkpoint code')
    half = torch.where(mag >= 16, mag-8, (mag >> 1) + ((mag & 1) & ((mag >> 1) & 1))).to(torch.uint8)
    packed = torch.zeros((n, g*128), dtype=torch.uint8)
    packed[:, :k] = (raw & 128) | half
    packed = packed.reshape(n, g, 128).permute(1, 0, 2).contiguous().view(torch.float8_e4m3fn)
    compensated = (scales * 2).contiguous()
    if not bool(torch.isfinite(compensated).all()):
        raise ValueError('native-half compensation overflows FP32 scale')
    if bias is None:
        bias = torch.zeros(n, dtype=torch.float32)
    if (bias.device.type != 'cpu' or bias.dtype != torch.float32 or tuple(bias.shape) != (n,)
            or not bool(torch.isfinite(bias).all())):
        raise ValueError('finite CPU FP32 bias [N] required')
    # Reject a BF16 plan before capture if scaled decoded weights can overflow.
    # This is load-time block metadata, not a per-operator device readback.
    maxima = packed.float().abs().amax(-1)
    padded_maxima = torch.zeros((g, ((n+127)//128)*128))
    padded_maxima[:, :n] = maxima
    maxima = padded_maxima.reshape(g, -1, 128).amax(-1).T
    safe32 = bool(torch.isfinite((maxima*compensated).bfloat16()).all())
    s16 = compensated.bfloat16().float()
    safe16 = bool(torch.isfinite(s16).all() and torch.isfinite((maxima*s16).bfloat16()).all())
    result = PreparedBlockFP8(packed, compensated, bias.contiguous().clone(), n, k,
        _hash(weight), _hash(scales), int(((mag < 16) & ((mag & 1) != 0)).sum()),
        bf16_fp32_scale_safe=safe32, bf16_bf16_scale_safe=safe16)
    result.validate()
    return result


def select_tp_row_group(weight, scales, *, rank, group_rows):
    """CPU slicing of checkpoint groups whose scale grids reset independently.

    MiMo QKV TP8 uses group_rows=(3392,)*8, NOT global ceil(N/128).
    Returns raw local tensors; preparation is a separate explicit action.
    """
    if (not group_rows or any(not isinstance(n, int) or n <= 0 for n in group_rows)
            or not isinstance(rank, int) or not 0 <= rank < len(group_rows)
            or weight.ndim != 2 or weight.shape[0] != sum(group_rows)
            or tuple(scales.shape) != (sum((n+127)//128 for n in group_rows), (weight.shape[1]+127)//128)):
        raise ValueError('checkpoint TP row-group metadata does not match tensors')
    row = sum(group_rows[:rank]); block = sum((n+127)//128 for n in group_rows[:rank])
    n = group_rows[rank]
    return weight[row:row+n].contiguous(), scales[block:block+(n+127)//128].contiguous()


def linear_block_fp8(x, prepared, *, activation, scale_math='fp32', bias=None, row_tile=None, reduce_op=None,
                     audit_tensors=None):
    """Build quant/batchMME/reduce OR decode/MME/finish in the current graph.

    `per_block_fp8` uses the original block128 amax/448 policy with native half
    adaptation. `bf16` keeps activation information and rounds scaled weights to
    BF16 transiently. `scale_math=bf16` is an explicit extra rounding option.
    SRAM placement must pass the compiled graph audit for each admitted shape.
    No implicit engine switch changes A16 to A8. No acquire/compile/mark_step.
    """
    import torch
    if not isinstance(prepared, PreparedBlockFP8):
        raise ValueError('PreparedBlockFP8 required; ordinary FP8 is incompatible')
    prepared.validate()
    if (x.ndim != 2 or x.shape[1] != prepared.k or x.dtype != torch.bfloat16
            or not x.is_contiguous() or x.device != prepared.weight.device):
        raise ValueError('contiguous BF16 [M,K] on the prepared weight device required')
    b = prepared.bias if bias is None else bias
    op = torch.ops.gaudi_block_fp8
    if activation == 'per_block_fp8':
        if row_tile is not None:
            raise ValueError('row_tile is an explicit BF16 MME plan only')
        if scale_math != 'fp32':
            raise ValueError('W8A8 block reduction uses FP32 scale arithmetic')
        q, sa = op.quant(x)
        return linear_block_fp8_quantized(q, sa, prepared, bias=b, reduce_op=reduce_op,
                                          audit_tensors=audit_tensors)
    if activation == 'bf16':
        if audit_tensors is not None:
            raise ValueError('staged FP32 audit applies only to explicit block A8')
        if reduce_op is not None:
            raise ValueError('A custom block reducer only applies to the explicit W8A8 path')
        if scale_math not in ('fp32', 'bf16'):
            raise ValueError('scale_math must be fp32 or bf16')
        if not (prepared.bf16_bf16_scale_safe if scale_math == 'bf16' else prepared.bf16_fp32_scale_safe):
            raise ValueError('scaled weights exceed finite BF16 decode domain; choose another explicit plan')
        decode = op.decode_fast if scale_math == 'bf16' else op.decode
        decoded = decode(prepared.weight, prepared.scales, prepared.k)
        if row_tile is not None:
            if not isinstance(row_tile, int) or not 0 < row_tile < x.shape[0]:
                raise ValueError('row_tile must split the known nonempty row shape')
            # One graph-local decoded tensor is shared. The compiler may clone
            # its producer; an actual graph audit must reject amplified weight
            # reads before this plan is admitted to production dispatch.
            return torch.cat(tuple(op.finish(op.mm(t, decoded), b)
                                   for t in op.split_rows(x, row_tile)), dim=0)
        p = op.mm(x, decoded)
        return op.finish(p, b)
    raise ValueError('activation must explicitly be bf16 or per_block_fp8')


def linear_block_fp8_quantized(q_native, activation_scales, prepared, bias=None, *, reduce_op=None,
                               audit_tensors=None):
    """Consume an already block-quantized activation, e.g. fused residual norm.

    q_native is native half-range FP8 [G,M,128]. FP32 scales are compensated by
    2 and have shape [G,M,1] (or [G,M], reshaped as a logical view). G=ceil(K/128).
    Quantization must have the same block128 OCP amax/448/RNE contract as quant;
    K-tail lanes must be zero. This entry never quantizes again.
    """
    import torch
    if not isinstance(prepared, PreparedBlockFP8):
        raise ValueError('PreparedBlockFP8 required')
    prepared.validate()
    if activation_scales.ndim == 2:
        activation_scales = activation_scales.unsqueeze(-1)
    if (q_native.dtype != torch.float8_e4m3fn or q_native.ndim != 3
            or q_native.shape[0] != (prepared.k+127)//128 or q_native.shape[2] != 128
            or activation_scales.dtype != torch.float32
            or tuple(activation_scales.shape) != (*q_native.shape[:2], 1)
            or q_native.device != prepared.weight.device
            or activation_scales.device != q_native.device):
        raise ValueError('native block128 activation/scale layout or device mismatch')
    p = torch.ops.gaudi_block_fp8.batch_mm(q_native, prepared.weight)
    reducer = torch.ops.gaudi_block_fp8.reduce if reduce_op is None else reduce_op
    b = prepared.bias if bias is None else bias
    y = reducer(p, activation_scales, prepared.scales, b)
    if audit_tensors is not None:
        # Diagnostic owner only. No readback, graph break or stream operation
        # occurs here; the production quality hook has disabled graph replay.
        audit_tensors.update(q_native=q_native, activation_scales=activation_scales,
                             prepared_weight=prepared.weight, prepared_scales=prepared.scales,
                             bias=b, partial=p, output=y)
    return y
