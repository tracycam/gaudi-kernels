#!/usr/bin/env python3
"""Compare the last two policies of a completed two-layer boundary fixture."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--guid', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = json.loads((args.case / 'result.json').read_text())
    assert result['status'] == 'DIAGNOSTIC' and result['candidate_accepted'] is False
    assert result['config']['hf_overrides']['num_hidden_layers'] == 2
    native = [r for r in result['runs'] if r['active']]
    control, candidate = native[-2:]
    assert control['ids'] == candidate['ids'] and len(candidate['ids']) == 16
    assert len(control['ranks']) == len(candidate['ranks']) == 8
    compared = 0
    for rank, (left, right) in enumerate(zip(control['ranks'], candidate['ranks'])):
        a = {v['position']: v for v in left['boundary_audit']}
        b = {v['position']: v for v in right['boundary_audit']}
        assert set(a) == set(b) == {4222, 4223, 4224, 4225}
        for position in a:
            values = []
            for row in (a[position], b[position]):
                path = args.case / 'boundary-audit' / Path(row['path']).name
                assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
                # Only our own source-sealed protocol4 tensor captures.
                values.append(torch.load(path, weights_only=False, map_location='cpu'))
            for key in ('hidden', 'selected', 'logits'):
                lhs, rhs = [v[key] for v in values]
                assert lhs.dtype == rhs.dtype and lhs.shape == rhs.shape
                assert torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8)), (rank, position, key)
                compared += 1
    kernels = []
    for rank in range(8):
        count = 0
        with (args.case / f'native-device-profile-rank{rank}.jsonl').open() as stream:
            for line in stream:
                event = json.loads(line)
                if event.get('ph') == 'B' and event.get('args', {}).get('op') == args.guid:
                    count += 1
        assert count > 0, (rank, 'candidate absent from device trace')
        kernels.append(dict(rank=rank, guid=args.guid, begin_event_count=count))
    report = dict(case=args.case.name, control=control['policy'], candidate=candidate['policy'],
                  all_cross_policy_boundary_tensor_bits_equal=True, compared_tensors=compared,
                  all_generated_ids_equal=True, actual_kernel_presence=kernels,
                  scope='two-layer interface/actual-kernel check; no model TPS or full-model quality acceptance')
    assert not args.out.exists(), 'refuse overwriting audit evidence'
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(compared_tensors=compared, kernel_observed_ranks=len(kernels), status='PASS')))


if __name__ == '__main__':
    main()
