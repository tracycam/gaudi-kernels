"""Check sealed full outputs and actual SRAM placement for the gather ABBA."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument('--archive', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--experiment', choices=('row-first', 'unroll8'), default='row-first')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((a.archive / 'remote-sha256.json').read_text())
cases = ['column-u4-v3-abba-0', 'row-u4-v3-abba-1',
         'row-u4-v3-abba-2', 'column-u4-v3-abba-3']
arms = ('column', 'row')
extras = ['row-u4-v3-profile-c', 'production-broadcast-vs-u4-row-v3-grid-d']
if a.experiment == 'unroll8':
    cases = ['u4-row-v3-abba-0', 'u8-row-v3-abba-1',
             'u8-row-v3-abba-2', 'u4-row-v3-abba-3']
    arms, extras = ('u4', 'u8'), []


def checked(path):
    expected = manifest['files'][str(path.relative_to(a.archive))]['sha256']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    return path


def tensors(value, prefix=''):
    for key, item in value.items():
        if isinstance(item, torch.Tensor):
            yield prefix + key, item
        elif isinstance(item, dict):
            yield from tensors(item, prefix + key + '.')


rows = []
for name in cases + extras:
    folder = a.archive / 'results' / name
    exit_record = json.loads(checked(folder / 'exit.json').read_text())
    assert exit_record['runner_exit_code'] == 0
    assert exit_record['postflight']['returncode'] == 0
    audit_path = a.out / (name + '-placement.json')
    subprocess.run([sys.executable, 'benchmarks/moe_route_tiles/audit.py',
                    str(folder), '--output', str(audit_path)], check=True)
    audit = json.loads(audit_path.read_text())
    assert audit['all_placement_pass']
    if name not in cases:
        continue
    d = json.loads(checked(folder / 'result.json').read_text())
    assert d['status'] == 'PASS_TP_LOCAL_MOE_FP32_BOUND'
    assert len(d['checks']) == 6 and all(x['metadata_equal'] for x in d['checks'])
    rows.append(dict(case=name, arm=arms[1] if name in cases[1:3] else arms[0],
                     event_us=statistics.median(x['event_us'] for x in d['timing']),
                     wall_us=statistics.median(x['wall_us'] for x in d['timing']),
                     samples=d['timing'], checks=d['checks'],
                     placement=[{k: r[k] for k in ('graph', 'compute_nodes', 'decoder_nodes',
                                'decoded_weight_SRAM_address_union_bytes', 'placement_pass')}
                                for r in audit['records']]))
comparisons = []
for state in ('uniform', 'hot', 'skew', 'zero', 'invalid', 'restored'):
    # Our own generated protocol-4 tensors, verified against the sealed digest.
    old = dict(tensors(torch.load(checked(a.archive / 'results' / cases[0] / (state + '.pt')),
                                 weights_only=False)))
    for name in cases[1:]:
        new = dict(tensors(torch.load(checked(a.archive / 'results' / name / (state + '.pt')),
                                     weights_only=False)))
        assert old.keys() == new.keys()
        for key, expected in old.items():
            actual = new[key]
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            x, y = [v.contiguous().view(torch.uint8) for v in (expected, actual)]
            assert torch.equal(x, y), (state, name, key)
            comparisons.append(dict(state=state, case=name, tensor=key,
                                    words=actual.numel(), dtype=str(actual.dtype),
                                    sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest()))
means = {arm: statistics.mean(r['event_us'] for r in rows if r['arm'] == arm)
         for arm in arms}
summary = dict(status='PASS_FULL_BITS_SRAM_AND_FP32_BOUND', arms=rows,
               event_means_of_arm_medians_us=means,
               latency_reduction_fraction=1 - means[arms[1]] / means[arms[0]],
               fp32_output_consumer_word_pairs=sum(x['words'] for x in comparisons
                                                  if x['tensor'] in ('y', 'consumer')),
               comparisons=comparisons,
               scope=('Complete synthetic TP-local MoE; same unroll4 decoder and metadata v3; '
                      if a.experiment == 'row-first' else
                      'Complete synthetic TP-local MoE; unroll4 vs8 with the same row-first gather and metadata v3; ')
                     + 'not model TPS or physical HBM utilization.')
(a.out / 'abba.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('arms', 'comparisons')}))
