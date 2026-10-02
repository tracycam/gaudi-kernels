"""Offline raw-bit parity and unfiltered whole-recipe ABBA summary."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def tensors(payload, prefix=''):
    for key, value in payload.items():
        name = prefix + key
        if isinstance(value, torch.Tensor):
            yield name, value
        elif isinstance(value, dict):
            yield from tensors(value, name + '.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cases = [a.results / f'bits-abba-{i}-{arm}'
             for i, arm in enumerate(('hoist', 'bits', 'bits', 'hoist'))]
    rows = []
    for case in cases:
        result = json.loads((case / 'result.json').read_text())
        exit_record = json.loads((case / 'exit.json').read_text())
        audit = json.loads((case / 'audit.json').read_text())
        assert result['status'] == 'PASS_TP_LOCAL_MOE_FP32_BOUND'
        assert exit_record['runner_exit_code'] == 0
        assert exit_record['postflight']['returncode'] == 0
        assert audit['all_placement_pass']
        assert len(result['timing']) == 4 and result['warmup_replays'] == 5
        assert len(result['checks']) == 6
        assert all(check['metadata_equal'] for check in result['checks'])
        rows.append(dict(case=case.name, arm=result['arm'],
                         timing=result['timing'],
                         event_us=statistics.median(t['event_us'] for t in result['timing']),
                         wall_us=statistics.median(t['wall_us'] for t in result['timing']),
                         checks=result['checks'],
                         placement=[{k:r[k] for k in ('graph','compute_nodes','decoder_nodes',
                             'workspace_bytes','decoded_weight_SRAM_address_union_bytes','placement_pass')}
                             for r in audit['records']]))
    comparisons = []
    for state in ('uniform', 'hot', 'skew', 'zero', 'invalid', 'restored'):
        reference = dict(tensors(torch.load(cases[0] / (state + '.pt'), weights_only=False)))
        for case in cases[1:]:
            actual = dict(tensors(torch.load(case / (state + '.pt'), weights_only=False)))
            assert actual.keys() == reference.keys()
            for name, expected in reference.items():
                got = actual[name]
                assert got.dtype == expected.dtype and got.shape == expected.shape
                # Invalid outputs are NaNs: compare storage, never floating equality.
                bad = int((got.contiguous().view(torch.uint8) !=
                           expected.contiguous().view(torch.uint8)).sum())
                row = dict(state=state, case=case.name, tensor=name,
                           dtype=str(got.dtype), shape=list(got.shape), elements=got.numel(),
                           mismatched_bytes=bad, reference_sha256=digest(expected),
                           actual_sha256=digest(got))
                comparisons.append(row)
                assert bad == 0, row
    control = [r for r in rows if r['arm'] == 'hoist']
    candidate = [r for r in rows if r['arm'] == 'bits']
    means = {metric:{arm:statistics.mean(r[metric] for r in values)
                     for arm, values in [('hoist',control),('bits',candidate)]}
             for metric in ('event_us','wall_us')}
    report = dict(status='PASS_FULL_RAW_BITS_AND_BOUND', arms=rows,
                  means_of_arm_medians=means,
                  latency_reduction_fraction=1-means['event_us']['bits']/means['event_us']['hoist'],
                  throughput_ratio=means['event_us']['hoist']/means['event_us']['bits'],
                  compared_output_fp32_cells=sum(r['elements'] for r in comparisons
                                               if r['tensor'] in ('y','consumer')),
                  comparisons=comparisons,
                  scope='Fresh-process same-GUID ABBA; every recorded trial retained. '
                        'Synthetic TP-local T513/E384/R8/C32/Ntile2048, metadata v3. '
                        'No full-model or production broadcast comparison.')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('arms','comparisons')}))


if __name__ == '__main__':
    main()
