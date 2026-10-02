"""Offline summary of preserved device gates, including failed candidates."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument('--archive', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
records = []
for path in sorted(a.archive.glob('module7-*/results/*/result.json')):
    result = json.loads(path.read_text())
    exit_record = json.loads((path.parent / 'exit.json').read_text())
    row = dict(case=str(path.parent.relative_to(a.archive)),
               source_commit=exit_record['git_commit'], status=result['status'],
               runner_exit_code=exit_record['runner_exit_code'],
               elapsed_s=exit_record['elapsed_s'],
               postflight_returncode=exit_record['postflight']['returncode'],
               all_numeric_pass=result.get('all_numeric_pass', False),
               direct=result['direct'],
               whole_chain_checks=result.get('whole_chain_checks', []),
               placement=result.get('placement'),
               timing=result['timing'], result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               error=result.get('error'))
    raw = path.parent / 'direct.pt'
    if raw.exists():
        row['direct_raw_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
        row['candidate_fp64'] = []
        # Our hash-verified harness fixtures use target-runtime pickle protocol 4;
        # the local 2.9 weights-only reader does not support opcode FRAME.
        # This entry accepts only the trusted, locally produced probe archive.
        for fixture in torch.load(raw, map_location='cpu', weights_only=False):
            for variant in ('p6', 'p4'):
                if variant not in fixture:
                    continue
                y, oracle, sumabs = fixture[variant].double(), fixture['oracle'], fixture['sumabs']
                error = (y-oracle).abs()
                row['candidate_fp64'].append(dict(
                    mode=fixture['mode'], variant=variant,
                    pairs=y.numel(), finite=bool(torch.isfinite(y).all()),
                    failures=int((error > 2e-6*sumabs + 1e-35).sum()),
                    max_abs=float(error.max()),
                    relative_l2=float(error.norm()/oracle.norm().clamp_min(1e-300)),
                    max_sumabs_backward_error=float((error/sumabs.clamp_min(1e-300)).max())))
    records.append(row)
summary = dict(all_pass=all(r['runner_exit_code'] == 0 and r['all_numeric_pass'] for r in records),
               records=records, device_used=True, analysis_device_used=False,
               model_quality_qualified=False, physical_HBM_measured=False,
               scope='All preserved public down-combine gates; a partial numerical pass never replaces whole-case failure',
               tolerance='Existing normal-domain diagnostic: abs(error) <= 2e-6*sumabs + 1e-35')
a.output.write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(dict(all_pass=summary['all_pass'], cases=[dict(case=r['case'], status=r['status'],
    candidate_fp64=r.get('candidate_fp64', [])) for r in records]), indent=2))
