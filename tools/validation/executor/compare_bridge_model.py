"""Compare ordinary-framework A/C runs; do not turn partial layers into TPS proof."""
import argparse
import json
from pathlib import Path


def compare(baseline, candidate):
    a = json.loads((baseline / 'result.json').read_text())
    c = json.loads((candidate / 'result.json').read_text())
    checks = {
        'completed': a['status'] == c['status'] == 'PASS',
        'same_layers': a['model_layers'] == c['model_layers'],
        'same_model_options': a['options'] == c['options'],
        'same_run_keys': [(r['batch'], r['repeat']) for r in a['runs']] ==
                         [(r['batch'], r['repeat']) for r in c['runs']],
        'eight_ranks': len(a.get('ranks', [])) == len(c.get('ranks', [])) == 8,
        'no_external_executors': all(not r['external_executors']
                                     for arm in (a, c) for r in arm.get('ranks', [])),
    }
    rows = []
    if checks['same_run_keys']:
        for old, new in zip(a['runs'], c['runs']):
            exact = old['outputs'] == new['outputs']
            rows.append(dict(batch=old['batch'], repeat=old['repeat'],
                exact_outputs=exact,
                baseline_tps=old['resident_tokens_per_second'],
                candidate_tps=new['resident_tokens_per_second'],
                speedup=new['resident_tokens_per_second'] / old['resident_tokens_per_second'],
                baseline_total_tps=old['total_tokens_per_second'],
                candidate_total_tps=new['total_tokens_per_second'],
                baseline_steps=old['resident_step_count'],
                candidate_steps=new['resident_step_count']))
    checks['exact_outputs'] = bool(rows) and all(row['exact_outputs'] for row in rows)
    checks['candidate_native_coverage'] = all(
        rank.get('plans') and all(p['ready'] and not p['failed'] and p['replays'] > 0
                                 for p in rank['plans']) for rank in c.get('ranks', []))
    qualified = all(checks.values()) and c['model_layers'] == 70
    b1 = [row for row in rows if row['batch'] == 1]
    return dict(status='PASS' if all(checks.values()) else 'FAIL',
        checks=checks, model_layers=c['model_layers'], full_model_qualified=qualified,
        b1_90_tps_observed=qualified and bool(b1) and all(row['candidate_tps'] >= 90 for row in b1),
        runs=rows, candidate_plans=[dict(rank=r['rank'], plans=r['plans']) for r in c['ranks']],
        scope='Same application/compute configuration, output equality and native coverage; '
              'resident TPS excludes prefill/warmup; total TPS includes them. '
              'Output equality is not independent model-quality validation. '
              'Fresh-process timing pairs are not same-process ABBA.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'full_model_qualified', 'b1_90_tps_observed', 'runs')}, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
