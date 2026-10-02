"""CPU-only numerical diagnostic; never a serving or performance backend.

The OCP case implements block128 amax*FP32(1/448), FP32 division, E4M3 RNE,
then FP64 MAC on quantized values and original block scales. The native case
adds the two explicit half-range RNE adaptations. Comparing these separates
activation quantization, native range adaptation, and device arithmetic.
The FP64 sum is an independent accuracy reference, not GPU bitwise emulation.
"""


def reference(x, weight, scales, *, native_half=False, bias=None):
    import torch
    if (x.device.type != 'cpu' or weight.device.type != 'cpu' or scales.device.type != 'cpu'
            or x.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn
            or scales.dtype != torch.float32 or x.ndim != 2 or weight.ndim != 2
            or x.shape[0] != 1 or x.shape[1] != weight.shape[1]):
        raise ValueError('CPU M1 BF16 activation, original OCP FP8 weight and FP32 scales required')
    n, k = weight.shape
    groups = (k + 127) // 128
    if tuple(scales.shape) != ((n + 127)//128, groups):
        raise ValueError('original block128 scale grid mismatch')
    if not bool(torch.isfinite(x.float()).all() and torch.isfinite(weight.float()).all()
                and torch.isfinite(scales).all() and (scales >= 0).all()):
        raise ValueError('finite original inputs required')
    a = torch.zeros((groups, 128), dtype=torch.float32)
    a.view(-1)[:k] = x.float().view(-1)
    sa = a.abs().amax(-1, keepdim=True).clamp_min(1e-10) * (1.0 / 448.0)
    qa = (a / sa).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    w = torch.zeros((n, groups * 128), dtype=torch.float32)
    w[:, :k] = weight.float()
    if native_half:
        qa = (qa * .5).to(torch.float8_e4m3fn).float()
        w = (w * .5).to(torch.float8_e4m3fn).float()
        sa = sa * 2
        scales = scales * 2
    # Quantized products are exact in FP64. Keep the source scale grid through
    # each K block; never replace it with channel scales or BF16 dequantization.
    wg = w.reshape(n, groups, 128).permute(1, 2, 0).double()
    partial = torch.bmm(qa.double().unsqueeze(1), wg).squeeze(1)
    sw = scales.repeat_interleave(128, dim=0)[:n].T.double()
    y = (partial * sa.double() * sw).sum(0, keepdim=True)
    if bias is not None:
        if bias.device.type != 'cpu' or bias.numel() != n:
            raise ValueError('CPU bias with N elements required')
        y = y + bias.double()
    return y.to(torch.bfloat16)
