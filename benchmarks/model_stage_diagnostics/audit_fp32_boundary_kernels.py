"""Independent full-tensor and executed-kernel checks for the two-layer gate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    d = json.loads((a.case / 'result.json').read_text())
    assert d['status'] == 'DIAGNOSTIC' and not d['candidate_accepted']
    assert d['quality_contract'] == 'fp32_arithmetic_v1'
    assert d['config']['hf_overrides']['num_hidden_layers'] == 2
    checks, compared = [], 0
    for policy in dict.fromkeys(r['policy'] for r in d['runs']):
        bridge = next(r for r in d['runs'] if r['policy'] == policy and not r['active'])
        native = next(r for r in d['runs'] if r['policy'] == policy and r['active'])
        assert len(bridge['ids']) == len(native['ids']) == 16 and bridge['ids'] == native['ids']
        assert len(bridge['ranks']) == len(native['ranks']) == 8
        for rank, (left, right) in enumerate(zip(bridge['ranks'], native['ranks'])):
            records = [{v['position']: v for v in r['boundary_audit']} for r in (left, right)]
            assert set(records[0]) == set(records[1]) == {4222, 4223, 4224, 4225}
            for pos in records[0]:
                tensors = []
                for record in records:
                    row = record[pos]
                    path = a.case / 'boundary-audit' / Path(row['path']).name
                    assert sha(path) == row['sha256']
                    # These are locally sealed files from our own model runner.
                    tensors.append(torch.load(path, weights_only=False, map_location='cpu'))
                for name in ('hidden', 'selected', 'logits'):
                    lhs, rhs = [t[name] for t in tensors]
                    assert lhs.shape == rhs.shape and lhs.dtype == rhs.dtype
                    assert torch.equal(lhs.contiguous().view(torch.uint8), rhs.contiguous().view(torch.uint8))
                    compared += 1
            checks.append(dict(policy=policy, rank=rank, positions=sorted(records[0]),
                               full_bridge_native_tensor_bits_equal=True))
    assert compared == 288
    guids = ('gk_pure_rmsnorm_bf16_v1', 'gk_residual_rmsnorm_bf16_v1',
             'gk_moe_gp_scale_tail_v1', 'gk_swa128_avgroup_group_v0')
    traces = []
    for rank in range(8):
        path = a.case / f'native-device-profile-rank{rank}.jsonl'
        counts = Counter()
        for line in path.open():
            event = json.loads(line)
            if event.get('ph') == 'B':
                counts[event.get('args', {}).get('op')] += 1
        assert all(counts[g] > 0 for g in guids)
        traces.append(dict(rank=rank, sha256=sha(path), executed_kernel_events={g: counts[g] for g in guids}))
    quality = d['fp32_quality']
    assert all(quality[k] for k in ('candidate_teacher_pass', 'staged_qkv_pass', 'matched_fp32_reference_pass'))
    assert not quality['normal_eos_and_tasks_pass'] and not quality['passed']
    a.out.parent.mkdir(parents=True, exist_ok=True)
    assert not a.out.exists()
    result = dict(case=a.case.name, result_sha256=sha(a.case / 'result.json'), checked_tensors=compared,
                  boundary_checks=checks, executed_kernels=traces, precision_preflight=quality,
                  scope='two-layer integration only; different norm policies are not required to match each other; no model TPS acceptance')
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(checked_tensors=compared, actual_kernel_ranks=len(traces), status='PASS_PREFLIGHT_ONLY')))


if __name__ == '__main__':
    main()
