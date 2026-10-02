from gaudi_kernels.engine.context import context as execution_context
'FP32 intermediates, original storage; no claim of exact real arithmetic.'
from pathlib import Path
import torch
torch.ops.load_library(execution_context().path('precision'))

@torch.library.register_fake('precision_fix::sum')
def _sum(x, ranks, bf16):
    return x.new_empty((x.shape[0] // ranks, x.shape[1]), dtype=torch.bfloat16 if bf16 else torch.float32)

@torch.library.register_fake('precision_fix::combine')
def _combine(p, r, d, h):
    return p.new_empty((r.shape[0], h), dtype=torch.float32)

@torch.library.register_fake('precision_fix::mm')
def _mm(x, w):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float32)

def combine(p, r, d, h, topk=None):
    return torch.ops.precision_fix.combine(p, r.float().clone(), d, h)

def sum_ranks(x, ranks, bf16):
    return torch.ops.precision_fix.sum(x, ranks, bf16)

def mm(x, w):
    return torch.ops.precision_fix.mm(x, w)
