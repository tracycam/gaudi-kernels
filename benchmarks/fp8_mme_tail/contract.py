"""Independent BF16-operand FP32 forward-error envelope for the padding ablation.

Not the block-A8 model contract: the serving A16 operands are BF16 decoded
weights, and padding must not change their representation or bias arithmetic.
"""
import math
import torch

PADS=(528,544,576,640,768)
def plans(pads=PADS):
    if not pads or any(type(m)is not int or m not in PADS for m in pads)or len(set(pads))!=len(pads):
        raise ValueError('unique fixed padding sizes 528/544/576/640/768 required')
    return [513,*pads]

def reference(x,w,bias):
    if x.device.type!='cpu' or w.device.type!='cpu' or x.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16:
        raise ValueError('independent CPU BF16 operands required')
    if x.ndim!=2 or w.ndim!=2 or x.shape[1]!=w.shape[1] or bias.shape!=(w.shape[0],) or bias.dtype!=torch.float32:
        raise ValueError('dot/bias geometry')
    if not all(bool(torch.isfinite(a).all())for a in (x,w,bias)):raise ValueError('nonfinite operand unsupported')
    # Every BF16 value is an integer multiple of 2**(frexp_exponent-8).
    # Their product lattice stays normal, so any cancellation is either zero
    # or normal FP32. This conservatively excludes unqualified FTZ domains.
    def unit(a):
        nz=a.float().abs();nz=nz[nz!=0]
        return int(torch.frexp(nz)[1].min())-(24 if a.dtype==torch.float32 else 8) if nz.numel()else 1000
    if unit(x)+unit(w)<-126 or (bool((bias!=0).any())and unit(bias)<-126):
        raise ValueError('subnormal intermediate lattice unsupported')
    xd,wd=x.double(),w.double();center=xd@wd.T+bias.double();absolute=xd.abs()@wd.abs().T+bias.double().abs()
    if not bool((absolute<torch.finfo(torch.float32).max).all()):raise ValueError('overflow envelope unsupported')
    k=x.shape[1];g32=(k+1)*2**-24/(1-(k+1)*2**-24);g64=(k+1)*2**-53/(1-(k+1)*2**-53)
    # Outward allowance includes the FP64 center and absolute-sum calculation.
    radius=torch.nextafter(absolute*(g32+g64)/(1-g64),torch.full_like(absolute,float('inf')))
    radius[absolute==0]=0
    return dict(center=center,radius=radius,absolute=absolute,operations=k+1,gamma32=g32,
                scope='BF16 products exact; arbitrary legal FP32 dot tree and FP32 bias, then BF16 RNE. No FP64 equality requirement.')

def assess(actual,ref):
    if actual.dtype!=torch.bfloat16 or actual.shape!=ref['center'].shape:raise ValueError('full BF16 output required')
    actual=actual.cpu();y=actual.double();finite=torch.isfinite(y)
    lo=(torch.nextafter(actual,torch.full_like(actual,-float('inf'))).double()+y)/2
    hi=(torch.nextafter(actual,torch.full_like(actual,float('inf'))).double()+y)/2
    low=ref['center']-ref['radius'];high=ref['center']+ref['radius']
    bad=~finite|(hi<low)|(lo>high)
    # Zero forward radius requires the independently rounded exact center.
    exact=ref['radius']==0;bad|=exact&(y!=ref['center'].bfloat16().double())
    return dict(pass_all=not bool(bad.any()),bad_cells=int(bad.sum()),cells=actual.numel(),
                max_absolute_difference=float((y-ref['center']).abs().max()),
                relative_l2_diagnostic=float((y-ref['center']).norm()/ref['center'].norm().clamp_min(1e-300)),
                max_fp32_radius=float(ref['radius'].max()),gamma32=ref['gamma32'],
                witness=bad.nonzero()[0].tolist()if bool(bad.any())else None)

def byte_ledger(m):
    return dict(M=m,semantic_M=513,extra_rows=m-513,semantic_flops=2*513*3392*6144,
        padded_flops=2*m*3392*6144,compute_ratio=m/513,
        original_weight_bytes=3392*6144,activation_padded_bf16_bytes=2*m*6144,
        padded_fp32_partial_bytes=4*m*3392,padded_bf16_output_bytes=2*m*3392,
        retained_output_bytes=2*513*3392,
        scope='Logical tensor sizes and useful/executed algebra, not physical transfers or compiler placement')
