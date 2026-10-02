"""Export public-op intermediates for a pinned equal-input QKV witness.

This diagnostic changes intermediate lifetimes. It must reproduce the observed
BF16 output before attribution; it is not a serving-placement or TPS benchmark.
Run only through the standard bounded, locked module runner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--row', type=Path, required=True)
p.add_argument('--witness', type=Path, required=True)
p.add_argument('--block-library', type=Path, required=True)
p.add_argument('--reduce-library', type=Path, required=True)
a = p.parse_args()
import torch
import habana_frameworks.torch.core as hc

assert os.environ['GAUDI_KERNELS_MODULE_ID'] == '7'
out = Path(os.environ['PROBE_OUT']) / 'intermediates'
out.mkdir(exist_ok=False)
expected = [(a.block_library, '3c9a092b4461c0990200cb88fb04aab62afe031b4e3b0ce488d0177aac6c6e0a'),
            (a.reduce_library, '4ea7ce7925c66a484167b0a3679e9475f7abf95d52ecbbd6ce7e762a4718b5e4')]
for path, digest in expected:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    torch.ops.load_library(str(path.resolve()))
identity = json.loads((a.row/'checkpoint-identity.json').read_text())
raw_w = (a.row/'weight-e4m3-shard.bin').read_bytes()
raw_s = (a.row/'scales-f32-shard.bin').read_bytes()
assert hashlib.sha256(raw_w).hexdigest() == identity['weight_shard_sha256']
assert hashlib.sha256(raw_s).hexdigest() == identity['scale_shard_sha256']
n, k = identity['group_rows'], identity['k']
assert (n, k) == (3392, 6144)
w = torch.frombuffer(bytearray(raw_w), dtype=torch.uint8).reshape(n, k)
s = torch.frombuffer(bytearray(raw_s), dtype=torch.float32).reshape((n+127)//128, k//128)
witness = torch.load(a.witness, map_location='cpu', weights_only=False)
assert (witness['record']['layer'], witness['record']['rank'], witness['record']['output_index']) == (
    identity['layer'], identity['rank'], identity['local_row'])
mag = (w & 127).to(torch.int16)
half = torch.where(mag >= 16, mag-8, (mag >> 1)+((mag & 1) & ((mag >> 1) & 1))).to(torch.uint8)
packed = ((w & 128) | half).reshape(n, k//128, 128).permute(1, 0, 2).contiguous()
x = witness['input_bf16'].to('hpu')
pw = packed.view(torch.float8_e4m3fn).to('hpu')
sw = (s*2).to('hpu')
bias = torch.zeros(n, dtype=torch.float32, device='hpu')
hc.mark_step(); torch.hpu.synchronize()

def apply():
    q, sa = torch.ops.gaudi_block_fp8.quant(x)
    partial = torch.ops.gaudi_block_fp8.batch_mm(q, pw)
    sequential = torch.ops.gaudi_block_fp8.reduce(partial, sa, sw, bias)
    compensated = torch.ops.gk_reduce_experiment.neumaier(partial, sa, sw, bias)
    return q, sa, partial, sequential, compensated

rows, held = [], []
for mode in ('eager', 'graph'):
    if mode == 'eager':
        values = apply(); hc.mark_step(); torch.hpu.synchronize()
    else:
        graph = torch.hpu.HPUGraph()
        with torch.hpu.graph(graph):
            values = apply()
        hc.mark_step(); torch.hpu.synchronize()
        graph.replay(); torch.hpu.synchronize(); held.append(graph)
    held.append(values)
    host = [v.detach().cpu() for v in values]
    for name, t in zip(('quant-fp8', 'activation-scales-f32', 'partial-f32',
                        'sequential-bf16', 'neumaier-bf16'), host):
        (out/f'{mode}-{name}.bin').write_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())
    y = host[-1]
    differences = int((y.view(torch.int16) != witness['actual_qkv'].view(torch.int16)).sum())
    rows.append(dict(mode=mode, different_bf16_outputs_vs_production=differences,
                     all_finite=all(bool(torch.isfinite(v).all()) for v in host[1:]),
                     witness_bits=int(y[0, identity['local_row']].view(torch.int16)) & 65535))
report = dict(rows=rows, witness=witness['record'], checkpoint_identity=identity,
              full_output_reproduced=all(r['different_bf16_outputs_vs_production'] == 0 for r in rows),
              scope=__doc__, model_acceptance=False)
(out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(rows, indent=2))
assert report['full_output_reproduced'] and all(r['all_finite'] for r in rows)
