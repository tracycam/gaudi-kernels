"""Device-free Meta checks of the actual public pure-RMS bridge."""
import argparse
import json
import os
os.environ.setdefault('PT_HPU_LAZY_MODE', '1')
import torch
import habana_frameworks.torch

p = argparse.ArgumentParser()
p.add_argument('--extension', required=True)
a = p.parse_args()
torch.ops.load_library(a.extension)
op = torch.ops.gaudi_kernels._pure_rmsnorm_bf16
passed = []
for m, h in [(1, 17), (2, 129), (8, 6144), (513, 8192)]:
    x = torch.empty((m, h), device='meta', dtype=torch.bfloat16)
    w = torch.empty((h,), device='meta', dtype=torch.bfloat16)
    y = op(x, w, 1e-6)
    assert y.shape == x.shape and y.dtype == torch.bfloat16
    passed.append([m, h])
x = torch.empty((1, 6144), device='meta', dtype=torch.bfloat16)
w = torch.empty((6144,), device='meta', dtype=torch.bfloat16)
bad = [(x.float(), w, 1e-6), (x, w.float(), 1e-6), (x, w[:128], 1e-6),
       (x.reshape(1, 1, 6144), w, 1e-6), (x[:, ::2], w, 1e-6),
       (x, w, 0.), (x, w, -1.), (x, w, float('nan')), (x, w, float('inf')),
       (torch.empty((1, 8193), device='meta', dtype=torch.bfloat16),
        torch.empty((8193,), device='meta', dtype=torch.bfloat16), 1e-6)]
for item in bad:
    try:
        op(*item)
    except RuntimeError:
        pass
    else:
        raise AssertionError('invalid Meta input accepted')
print(json.dumps(dict(status='PASS_META_CPU_ONLY', valid=passed, rejected=len(bad))))
