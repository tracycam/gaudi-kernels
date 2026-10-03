"""Broad FP32 forward-error checks, not a precision/quality selection gate.

For the qualified finite MXFP4 GP inputs, BF16 activations and dyadic weights
have exactly representable FP32 products. Bound both addition trees against
their absolute product sum. This accepts cancellation and legal FP32 ordering
while detecting the wrong-lane reference that motivated this check.
"""


def dot_error_bound(products):
    import torch
    if products.dtype != torch.float32 or products.ndim != 2:
        raise ValueError('FP32 product rows required')
    k = products.shape[-1]
    operations = k + (k+31)//32
    u = torch.finfo(torch.float32).eps/2
    if operations*u >= .5:
        raise ValueError('Dot extent outside diagnostic bound')
    gamma = operations*u/(1-operations*u)
    # Four gamma factors conservatively cover two different reductions,
    # scaling operations and rounding in the FP32 absolute-sum measurement.
    return 4*gamma*products.abs().sum(-1) + 4*k*torch.finfo(torch.float32).tiny
