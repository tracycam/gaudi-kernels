"""Separate measured MME, factor, product and accumulation rounding errors.

CPU emulation is bound to the existing reducer's source instruction order.
Factor compensation is an explicit untested alternative, not a device result.
"""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--row', type=Path, required=True)
p.add_argument('--witness', type=Path, required=True)
p.add_argument('--device', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(1)
witness = torch.load(a.witness, map_location='cpu', weights_only=False)['record']
partial64 = np.fromfile(a.row/'fp64-partials.bin', np.float64)
sa = np.fromfile(a.row/'cpu-activation-scales-f32.bin', np.float32)
sw = np.fromfile(a.row/'native-weight-scales-f32.bin', np.float32)
q = np.fromfile(a.row/'cpu-quantized-activation-fp8.bin', np.uint8)
factors = (sa*sw).astype(np.float32)
bounds = json.loads((a.row/'cpu-reconstruction.json').read_text())['bounds']
libm = ctypes.CDLL('libm.so.6')
fma = libm.fmaf
fma.argtypes = [ctypes.c_float]*3
fma.restype = ctypes.c_float
f32 = np.float32

def bf(value):
    return dict(value=float(value), bits=int(torch.tensor(float(value), dtype=torch.float64)
                                            .bfloat16().view(torch.int16)) & 65535)

def sum64(partial, scale):
    return float((torch.from_numpy(partial).double()*torch.from_numpy(scale).double()).sum())

def neumaier(partial, factor_correction=False):
    total, correction = f32(0), f32(0)
    for v, factor, x, w in zip(partial, factors, sa, sw):
        term = f32(v*factor)
        product_error = f32(fma(float(v), float(factor), -float(term)))
        if factor_correction:
            scale_error = f32(fma(float(x), float(w), -float(factor)))
            product_error = f32(fma(float(v), float(scale_error), float(product_error)))
        updated = f32(total+term)
        residual = (f32(f32(total-updated)+term) if abs(total) >= abs(term)
                    else f32(f32(term-updated)+total))
        correction = f32(correction+f32(residual+product_error))
        total = updated
    return f32(total+correction)

exact_scale = sa.astype(np.float64)*sw.astype(np.float64)
reference = sum64(partial64, exact_scale)
assert bf(reference)['bits'] == witness['reference_bits']
midpoint = (witness['reference_value']+witness['actual_value'])/2
records = []
captured = []
for mode in ('eager', 'graph'):
    quant = np.fromfile(a.device/f'{mode}-quant-fp8.bin', np.uint8)
    scales = np.fromfile(a.device/f'{mode}-activation-scales-f32.bin', np.float32)
    partial_all = np.fromfile(a.device/f'{mode}-partial-f32.bin', np.float32).reshape(48, 1, 3392)
    partial = partial_all[:, 0, witness['output_index']].copy()
    output = np.fromfile(a.device/f'{mode}-neumaier-bf16.bin', np.uint16)
    sequential = np.fromfile(a.device/f'{mode}-sequential-bf16.bin', np.uint16)
    assert np.array_equal(q, quant)
    assert np.array_equal(scales.view(np.uint32), sa.view(np.uint32))
    actual = neumaier(partial)
    assert bf(actual)['bits'] == int(output[witness['output_index']]) == witness['actual_bits']
    only_mme = sum64(partial, exact_scale)
    with_factor_rounding = sum64(partial, factors)
    alternative = neumaier(partial, True)
    trees = {}
    for path in a.row.glob('partial-*.bin'):
        if path.stem.startswith('partial-device'):
            continue
        values = np.fromfile(path, np.float32)
        if len(values) == 48:
            trees[path.stem[8:]] = dict(
                different_bits_vs_device=int(np.count_nonzero(values.view(np.uint32) != partial.view(np.uint32))),
                neumaier_output=bf(neumaier(values)))
    exact_bound_violations = [g for g in range(48) if bounds[g]['all_orders_fp32_exact']
                              and partial[g] != partial64[g]]
    assert not exact_bound_violations, 'A measured partial violates the all-orders FP32 exactness bound'
    records.append(dict(mode=mode, activation_bytes_and_scales_bitwise=True,
        actual_neumaier_instruction_order_reproduced=True,
        reference=bf(reference), mme_only=bf(only_mme), mme_plus_factor_rounding=bf(with_factor_rounding),
        actual_neumaier=bf(actual), sequential_bits=int(sequential[witness['output_index']]),
        fp32_factor_compensation_cpu_only=bf(alternative),
        deltas=dict(mme=only_mme-reference, factor_product_rounding=with_factor_rounding-only_mme,
                    compensated_accumulation=float(actual)-with_factor_rounding),
        bf16_midpoint=midpoint,
        reference_midpoint_distance_in_fp32_ulps=(reference-midpoint)/abs(float(np.spacing(f32(midpoint)))),
        different_mme_partials_vs_fp64=int(np.count_nonzero(partial.astype(np.float64) != partial64)),
        different_mme_partials_vs_round_once_fp32=int(np.count_nonzero(partial.view(np.uint32) != partial64.astype(np.float32).view(np.uint32))),
        diagnostic_cpu_trees=trees, all_orders_exact_bound_violations=exact_bound_violations,
        groups=[dict(group=g, partial64=float(partial64[g]), actual_partial=float(partial[g]),
                     scale_product_exact=float(exact_scale[g]), scale_product_f32=float(factors[g]),
                     scale_product_residual=float(fma(float(sa[g]), float(sw[g]), -float(factors[g]))))
                for g in range(48)]))
    captured.append((quant, scales, partial_all, output, sequential))
assert all(np.array_equal(x.view(np.uint8), y.view(np.uint8)) for x, y in zip(*captured))
report = dict(witness=witness, rows=records, eager_graph_all_bits_equal=True,
              original_serving_mme_tree_identified=False,
              factor_compensation_device_tested=False, model_gate_changed=False,
              note='Probe exports intermediates and changes graph lifetimes; its full BF16 output matches the saved production output.')
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k != 'groups'} for r in records], indent=2))
