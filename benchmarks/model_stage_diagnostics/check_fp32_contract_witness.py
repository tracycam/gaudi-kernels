#!/usr/bin/env python3
"""Recheck all columns of the sealed real L0/R2 witness without any device API."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from gaudi_kernels.block_fp8 import prepare_block_fp8
from gaudi_kernels.block_fp8_fp32_contract import audit, build_reference, require_fast_backend


def load(path, dtype, shape):
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype).reshape(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    backend = require_fast_backend()
    source = args.repo / 'artifacts/builds/model-stage-diagnostics/70-b-layer0-rank2'
    row = source / 'row'
    device = source / 'device-a/results/qkv-l0-r2-a/intermediates'
    identity = json.loads((row / 'checkpoint-identity.json').read_text())
    for name, key in [('weight-e4m3-shard.bin', 'weight_shard_sha256'), ('scales-f32-shard.bin', 'scale_shard_sha256')]:
        assert hashlib.sha256((row / name).read_bytes()).hexdigest() == identity[key]
    x = load(row / 'input-bf16.bin', torch.bfloat16, (1, 6144))
    weight = load(row / 'weight-e4m3-shard.bin', torch.uint8, (3392, 6144)).view(torch.float8_e4m3fn)
    scales = load(row / 'scales-f32-shard.bin', torch.float32, (27, 48))
    start = time.monotonic()
    reference = build_reference(x, weight, scales)
    reference_s = time.monotonic() - start
    # The old probe uploaded these from its frozen producer but did not dump
    # the prepared weight back from HPU. Label this provenance limitation;
    # future live v1 audits read it back and bind the loaded implementation.
    prepared = prepare_block_fp8(weight, scales)
    records = []
    for mode in ('eager', 'graph'):
        actual = dict(q_native=load(device / (mode + '-quant-fp8.bin'), torch.uint8, (48, 1, 128)),
                      activation_scales=load(device / (mode + '-activation-scales-f32.bin'), torch.float32, (48, 1, 1)),
                      prepared_weight=prepared.weight, prepared_scales=prepared.scales, bias=prepared.bias,
                      partial=load(device / (mode + '-partial-f32.bin'), torch.float32, (48, 1, 3392)))
        for suffix, reduction in [('sequential', 'sequential'), ('neumaier', 'neumaier_fp32')]:
            actual['output'] = load(device / (mode + '-' + suffix + '-bf16.bin'), torch.bfloat16, (1, 3392))
            start = time.monotonic()
            report = audit(reference, actual, reduction=reduction)
            records.append(dict(mode=mode, reduction=reduction, diagnostic_wall_s=time.monotonic() - start, report=report))
    result = dict(scope=__doc__, source_checkpoint_identity=identity, reference_s=reference_s, cpu_backend=backend,
                  actual_partial_cells=48 * 3392, output_columns=3392, records=records,
                  all_numerical_gates_pass=all(r['report']['numerical_passed'] for r in records),
                  implementation_qualification=False, model_acceptance=False,
                  limitation='prepared bytes reconstructed from original uploaded inputs; old probe lacks prepared-weight device readback and live v1 implementation/frame binding; never backfill old model acceptance')
    inputs = [row / 'input-bf16.bin', row / 'weight-e4m3-shard.bin', row / 'scales-f32-shard.bin', row / 'checkpoint-identity.json']
    inputs += sorted(device.glob('*.bin'))
    result['source_files'] = {str(p.relative_to(args.repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'source_files', 'source_checkpoint_identity')}, indent=2))
    for record in records:
        print(record['mode'], record['reduction'], record['report']['classification'], record['report'].get('stages', {}).get('partial'))
    assert result['all_numerical_gates_pass']


if __name__ == '__main__':
    main()
