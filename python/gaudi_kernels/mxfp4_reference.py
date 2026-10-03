"""CPU FP32 reference for ragged W4A16 and K32 MXFP8 W4A8 MoE.

Not a serving backend or an HPU performance model. Canonical checkpoint bytes
use low/high nibbles for consecutive K values, unlike HistoricalN512 prepack.
The two policies share routing and nonlinear boundaries, not activation values.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MXFP4Weights:
    packed: torch.Tensor  # uint8 [E,N,ceil(K/2)]
    scales: torch.Tensor  # E8M0 uint8 [E,N,ceil(K/32)]
    k: int

    def validate(self):
        if type(self.k) is not int or self.k < 1:
            raise ValueError('positive logical K required')
        if (self.packed.device.type != 'cpu' or self.scales.device.type != 'cpu'
                or self.packed.dtype != torch.uint8 or self.scales.dtype != torch.uint8
                or self.packed.ndim != 3 or min(self.packed.shape) < 1
                or self.packed.shape[-1] != (self.k+1)//2
                or tuple(self.scales.shape) != (*self.packed.shape[:2], (self.k+31)//32)):
            raise ValueError('canonical CPU packed MXFP4 geometry required')
        if (self.scales == 255).any():
            raise ValueError('E8M0 NaN scale is not a finite-weight contract')

    def codes(self, expert):
        p = self.packed[expert]
        codes = torch.stack((p & 15, p >> 4), dim=-1).flatten(-2)[..., :self.k]
        lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                            -0., -.5, -1., -1.5, -2., -3., -4., -6.])
        return lut[codes.long()]


def quantize_mxfp8(x):
    """GPU-style K32 E4M3FN/E8M0, FP32 scale math, zero-padded K tail.

    Returns quantized float8 codes and uint8 scales. No native-Gaudi FP8
    conversion is implied. Native rebasing needs its own representability test.
    """
    if x.device.type != 'cpu' or x.ndim != 2 or x.shape[1] < 1:
        raise ValueError('CPU [M,K] reference input required')
    v = x.float()
    if not torch.isfinite(v).all():
        raise ValueError('nonfinite activation')
    k = v.shape[1]
    blocks = F.pad(v, (0, (-k) % 32)).reshape(v.shape[0], (k+31)//32, 32)
    amax = blocks.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = (torch.ceil(torch.log2(amax / 448.)) + 127.).clamp(0, 254)
    scale = torch.exp2(exponent - 127.)
    q = (blocks / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    if not torch.isfinite(q.float()).all():
        raise ValueError('activation exceeds finite MXFP8 reference domain')
    return q.reshape(v.shape[0], ((k+31)//32)*32)[:, :k], exponent.to(torch.uint8)


def linear_reference(x, weights, expert, *, policy):
    """K32 partial dot, scale, then FP32 accumulation; no FP64 equality gate.

    This defines a numerical reference, not mandatory hardware block boundaries.
    BF16 decode/MME variants may legally use a different FP32 accumulation tree.
    """
    weights.validate()
    if (x.device.type != 'cpu' or x.dtype != torch.bfloat16 or x.ndim != 2
            or x.shape[1] != weights.k or not torch.isfinite(x).all()
            or type(expert) is not int or not 0 <= expert < weights.packed.shape[0]):
        raise ValueError('finite CPU BF16 [M,K] and valid expert required')
    if policy not in ('w4a16', 'w4a8_mxfp8'):
        raise ValueError('explicit w4a16 or w4a8_mxfp8 required')
    if policy == 'w4a8_mxfp8':
        q, scales = quantize_mxfp8(x)
        a = q.float()
        sa = torch.exp2(scales.float()-127.)
    else:
        a = x.float()
        sa = torch.ones((x.shape[0], (weights.k+31)//32))
    w = weights.codes(expert)
    sw = torch.exp2(weights.scales[expert].float()-127.)
    out = torch.zeros((x.shape[0], w.shape[0]), dtype=torch.float32)
    for group, begin in enumerate(range(0, weights.k, 32)):
        # Keep both scales inside the K sum. A single post-GEMM scale is
        # generally wrong because scales vary with group, output and input row.
        partial = a[:, begin:begin+32] @ w[:, begin:begin+32].T
        out = out + (partial * sw[:, group]) * sa[:, group, None]
    if not torch.isfinite(out).all():
        raise ValueError('projection exceeds finite FP32 reference domain')
    return out


def moe_reference(x, ids, routing, gp, down, *, policy):
    """Actual routes, any E/M within CPU resources, GP/gate/down/combine.

    GP columns are canonical [gate,up]. Masked routes use ID -1 and zero
    routing weight. Duplicate valid IDs in one token are rejected explicitly.
    Grouping here is an offline oracle; no host route read is allowed in serving.
    """
    gp.validate()
    down.validate()
    e, two_h = gp.packed.shape[:2]
    if (two_h % 2 or down.packed.shape[:2] != (e, gp.k)
            or down.k != two_h//2):
        raise ValueError('GP/down MoE geometry mismatch')
    if (x.device.type != 'cpu' or x.ndim != 2 or x.shape[1] != gp.k
            or x.dtype != torch.bfloat16 or not torch.isfinite(x).all()
            or ids.device.type != 'cpu' or ids.dtype not in (torch.int32, torch.int64)
            or ids.ndim != 2 or ids.shape[0] != x.shape[0] or ids.shape[1] < 1
            or routing.device.type != 'cpu' or routing.dtype != torch.float32
            or routing.shape != ids.shape or not torch.isfinite(routing).all()):
        raise ValueError('CPU BF16 activation, integer routes, FP32 routing required')
    if ((ids < -1) | (ids >= e)).any() or (routing[ids == -1] != 0).any():
        raise ValueError('invalid or nonzero masked route')
    if policy not in ('w4a16', 'w4a8_mxfp8'):
        raise ValueError('explicit activation policy required')
    for row in ids:
        valid = row[row >= 0]
        if valid.unique().numel() != valid.numel():
            raise ValueError('duplicate expert in one token')
    partial = torch.zeros((*ids.shape, gp.k), dtype=torch.float32)
    for expert in ids[ids >= 0].unique().tolist():
        tokens, slots = (ids == expert).nonzero(as_tuple=True)
        gu = linear_reference(x[tokens], gp, expert, policy=policy).to(torch.bfloat16)
        gate, up = gu.chunk(2, dim=-1)
        hidden = F.silu(gate.float()).to(torch.bfloat16) * up
        partial[tokens, slots] = linear_reference(hidden, down, expert, policy=policy)
    out = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        out = out + partial[:, slot] * routing[:, slot, None]
    if not torch.isfinite(out).all():
        raise ValueError('combine exceeds finite FP32 reference domain')
    return out
