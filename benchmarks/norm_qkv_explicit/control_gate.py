"""CPU metadata/protocol gate only; no emulation of TPC arithmetic or HPU graph."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels import norm_qkv_explicit as api

events = []
x = torch.empty((1, 6144), dtype=torch.bfloat16, device='meta')
r = torch.empty_like(x)
g = torch.empty(6144, dtype=torch.bfloat16, device='meta')
prepared = SimpleNamespace(n=3392, k=6144)
reducer = object()
seen_payloads = []


def producer(x, residual, gamma, epsilon):
    events.append('fused_norm_quant')
    rr = torch.empty_like(x)
    q = torch.empty((48, 1, 128), dtype=torch.float8_e4m3fn, device='meta')
    scales = torch.empty((48, 1, 1), dtype=torch.float32, device='meta')
    seen_payloads.append((rr, q, scales))
    return rr, q, scales


def consumer(q, scales, weight, bias, *, reduce_op, audit_tensors):
    events.append('block_mm_reduce')
    assert q is seen_payloads[-1][1] and scales is seen_payloads[-1][2]
    assert weight is prepared and reduce_op is reducer
    return torch.empty((1, 3392), dtype=torch.bfloat16, device='meta')


api.residual_rmsnorm_block128_a8 = producer
api.linear_block_fp8_quantized = consumer
for _ in range(2):
    rr, y = api.residual_norm_qkv(x, r, g, 1e-6, prepared, norm_policy='fp32',
        activation_policy='per_block_fp8', reduce_op=reducer)
    assert rr is seen_payloads[-1][0] and y.shape == (1, 3392)
assert events == ['fused_norm_quant', 'block_mm_reduce']*2
assert seen_payloads[0][1] is not seen_payloads[1][1]
records = []
for reason, changes in [
    ('norm_policy', dict(norm_policy='vendor')),
    ('activation_policy', dict(activation_policy='bf16')),
    ('fused_producer_audit_witness_unqualified', dict(same_input_audit_active=True)),
    ('residual_m1_bf16_6144', dict(residual=None)),
    ('input_m1_bf16_6144', dict(x=torch.empty((8, 6144), dtype=torch.bfloat16, device='meta'))),
    ('input_m1_bf16_6144', dict(x=torch.empty((513, 6144), dtype=torch.bfloat16, device='meta'))),
    ('input_m1_bf16_6144', dict(x=torch.empty((1, 1, 6144), dtype=torch.bfloat16, device='meta'))),
    ('input_m1_bf16_6144', dict(x=x.float())),
    ('gamma_bf16_6144', dict(gamma=g.float())),
    ('epsilon', dict(epsilon=float('nan'))),
    ('epsilon_fp32_underflow', dict(epsilon=1e-100)),
]:
    kwargs = dict(x=x, residual=r, gamma=g, epsilon=1e-6, prepared=prepared,
                  norm_policy='fp32', activation_policy='per_block_fp8', reduce_op=reducer)
    kwargs.update(changes)
    before = len(events)
    try:
        api.residual_norm_qkv(**kwargs)
    except ValueError as exc:
        assert reason in str(exc)
    else:
        raise AssertionError('unsupported path accepted')
    assert len(events) == before
    records.append(dict(reason=reason, operations_before_rejection=0))
report = dict(status='PASS_CPU_STRUCTURE_ONLY', device_verified=False,
    serving_installer=False, events=events, rejection_cases=records,
    scope='explicit tensor ownership, reducer forwarding, no requantization, rejection before producer; not arithmetic validation')
(a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
