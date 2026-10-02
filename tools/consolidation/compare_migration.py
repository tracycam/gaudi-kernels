#!/usr/bin/env python3
"""Strict migration acceptance: full-model gates, byte hashes, identical-policy ABBA."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def quantile(values, fraction):
    values = sorted(values)
    location = fraction * (len(values) - 1)
    lo = int(location)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (location - lo)


def steady(row):
    steps = row['steps']
    indices = [i for i, s in enumerate(steps) if 32 <= s['emitted'] < 160]
    if not indices:
        return []
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError('Noncontiguous mature decode window')
    return steps[indices[0]:indices[-1] + 1]


def timing(rows):
    windows = [steady(r) for r in rows]
    if any(not w or sum(s['new_tokens'] for s in w) != 128
           or any(s['new_tokens'] not in (0, 1) for s in w) for w in windows):
        raise ValueError('Expected complete 128-token matured windows; stalled steps must remain')
    milliseconds = [s['ms'] for window in windows for s in window]
    duration = sum((w[-1]['end_ns'] - w[0]['begin_ns']) / 1e9 for w in windows)
    return {'tps': sum(s['new_tokens'] for w in windows for s in w) / duration,
        'p50_ms': quantile(milliseconds, .5), 'p90_ms': quantile(milliseconds, .9),
        'p99_ms': quantile(milliseconds, .99), 'steps': len(milliseconds)}


def layer_hashes(result):
    candidates = result['quality_plan']['candidate_policies']
    records = {}
    for row in result['teacher_forced']['rows']:
        if row['policy'] not in candidates:
            continue
        for rank in row['same_input_qkv_audit']:
            for audit in rank['same_input_qkv_audit']:
                key = (row['policy'], row['position'], rank['rank'], audit['layer'])
                if key in records:
                    raise ValueError('Duplicate per-layer audit')
                stages = audit['fp32_contract']['stages']
                required = {'q_native', 'activation_scales', 'prepared_weight', 'prepared_scales',
                            'bias', 'partial', 'epilogue'}
                if not required.issubset(stages) or audit['fp32_contract'].get('numerical_passed') is not True:
                    raise ValueError('Missing or rejected staged FP32 arithmetic evidence')
                if audit['plain_vs_staged'].get('all_bits_equal') is not True:
                    raise ValueError('Intermediate-retaining diagnostic changed the returned QKV output')
                records[key] = {'input': audit['input_sha256'],
                    'plain_output': audit['plain_vs_staged']['plain_sha256'],
                    'staged_output': audit['plain_vs_staged']['staged_sha256'],
                    **{name: entry['actual_sha256'] for name, entry in stages.items()}}
                if any(not isinstance(h, str) or not re.fullmatch(r'[0-9a-f]{64}', h) for h in records[key].values()):
                    raise ValueError('Invalid byte hash in per-layer evidence')
    expected = {(policy, position, rank, f'language_model.model.layers.{layer}.self_attn.qkv_proj')
        for policy in candidates for position in (0, 16, 48) for rank in range(8) for layer in range(70)}
    if set(records) != expected:
        raise ValueError(f'Incomplete 70-layer/8-rank/3-position hashes: {len(records)} vs {len(expected)}')
    return records


def compare(baseline_path, candidate_path, *, tensor_logits=True):
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if any(r.get('status') != 'PASS' or r.get('candidate_accepted') is not True for r in (baseline, candidate)):
        raise ValueError('Both full-model quality gates must pass before migration acceptance')
    if baseline['config'] != candidate['config']:
        raise ValueError('Model shape, dtype, parallelism or KV allocation config changed')
    old_runs = {r['name']: r for r in baseline['runs']}
    new_runs = {r['name']: r for r in candidate['runs']}
    if set(old_runs) != set(new_runs):
        raise ValueError('Run coverage differs')
    tokens = all(old_runs[name]['ids'] == row['ids'] for name, row in new_runs.items())
    old_hashes, new_hashes = layer_hashes(baseline), layer_hashes(candidate)
    mismatch = [list(key) for key in old_hashes if old_hashes[key] != new_hashes[key]]
    logits = []
    if tensor_logits:
        import torch
        for path in sorted((baseline_path.parent / 'quality-logits').glob('*.pt')):
            target = candidate_path.parent / 'quality-logits' / path.name
            if not target.is_file():
                raise ValueError('Missing complete logits file: ' + path.name)
            # These are our generated, locally preserved tensor artifacts.
            # The benchmark saved pickle protocol4; this pinned Torch version
            # cannot read protocol4 with its weights-only unpickler. Do not
            # apply this loader to untrusted external checkpoints.
            old = torch.load(path, weights_only=False, map_location='cpu')
            new = torch.load(target, weights_only=False, map_location='cpu')
            if (not isinstance(old, dict) or not isinstance(new, dict)
                    or not all(isinstance(t, torch.Tensor) for t in [*old.values(), *new.values()])):
                raise ValueError('Generated logits artifact is not a tensor-only dictionary')
            if set(old) != set(new):
                raise ValueError('Logits tensor fields changed')
            checks = {}
            hashes = {}
            for key in old:
                a, b = old[key], new[key]
                checks[key] = (a.shape == b.shape and a.dtype == b.dtype
                    and bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))))
                # Tensor bytes, not the nondeterministic .pt ZIP header.
                hashes[key] = {'baseline': hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                               'candidate': hashlib.sha256(b.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            logits.append({'file': path.name, 'checks': checks, 'hashes': hashes, 'pass': all(checks.values())})
        if not logits:
            raise ValueError('No complete logits comparisons')
    arms = candidate.get('migration_abba', [])
    if [r['arm'] for r in arms] != [0, 1, 2, 3]:
        raise ValueError('Missing same-process migration ABBA')
    if [r['installer'] for r in arms] != ['production_policy_reference', 'production_policy',
                                         'production_policy', 'production_policy_reference']:
        raise ValueError('Wrong old/new installer pairing')
    if not all(r['resolved_selection'] == arms[0]['resolved_selection'] for r in arms):
        raise ValueError('ABBA compared different arithmetic choices')
    paired = sorted([r for r in candidate['runs'] if 'native-policy-abba-' in r['name']],
                    key=lambda r: int(r['name'].rsplit('-', 1)[-1]))
    if len(paired) != 4 or not all(r.get('performance_qualified') for r in paired):
        raise ValueError('Unqualified paired timing')
    old_time, new_time = timing([paired[0], paired[3]]), timing([paired[1], paired[2]])
    performance = new_time['tps'] >= old_time['tps'] * .99 and new_time['p99_ms'] <= old_time['p99_ms']
    return {'pass': tokens and not mismatch and tensor_logits and all(r['pass'] for r in logits) and performance,
        'token_runs_checked': len(new_runs), 'tokens_bitwise': tokens,
        'qkv_layer_records_checked': len(old_hashes), 'qkv_layer_hash_mismatches': mismatch,
        'complete_logits': logits, 'same_process_abba': {'historical_installer': old_time, 'typed_installer': new_time,
            'tps_delta_percent': 100 * (new_time['tps'] / old_time['tps'] - 1),
            'pass': performance, 'rule': 'TPS >= 99% of paired control and p99 no worse'},
        'scope': 'Equivalent installer migration; QKV hashes cover all70 layers/8ranks/3positions. '
                 'Other intermediate layer boundaries are not hashed by this historical harness; '
                 'this is not completion of the full architecture migration.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.baseline, a.candidate)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'complete_logits'}, indent=2))
    raise SystemExit(0 if result['pass'] else 1)
