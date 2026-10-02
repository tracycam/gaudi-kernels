"""Read our frozen model case and seal a small, honest evidence summary.

Only run on the project's own trusted experiment directory: logits were saved
with Torch 2.11's pickle protocol, which older weights_only readers cannot load.
This tool never changes source experiment files or upgrades failed acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = json.loads((a.case / 'result.json').read_text())
    first = result['runs'][0]['ids']
    rows = []
    for run in result['runs']:
        rows.append({'name': run['name'], 'timing': run.get('serving_timing', run.get('diagnostic_timing')),
            'retrieval_pass': run['quality_pass'], 'within_policy_token_match': run['bitwise_token_match'],
            'first_difference_from_first_policy': next((i for i, (x, y) in enumerate(zip(first, run['ids'])) if x != y), None),
            'raw_timing_is_not_quality_acceptance': True})
    report = {'case': a.case.name, 'status': result['status'],
              'candidate_accepted': result['candidate_accepted'], 'runs': rows,
              'teacher_forced': result.get('teacher_forced'),
              'same_contract_reference': result.get('same_contract_reference'),
              'chat_quality': result.get('chat_quality'),
              'arithmetic_preserving_gp_pairs': result.get('arithmetic_preserving_gp_pairs'),
              'batch_runs': [{k: v for k, v in row.items() if k not in ('steps', 'ids', 'texts')}
                             for row in result.get('batch_runs', [])],
              'long_context': {k: v for k, v in result.get('long_context', {}).items() if k != 'runs'},
              'long_context_runs': [{k: v for k, v in row.items()
                                     if k not in ('steps', 'ids', 'text', 'ranks', 'controls')}
                                    for row in result.get('long_context', {}).get('runs', [])],
              'scope': 'bounded experiment, no extrapolated TPS or broad quality claim'}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / (a.case.name + '-summary.json')).write_text(json.dumps(report, indent=2) + '\n')
    manifest = []
    for path in sorted(a.case.rglob('*')):
        if path.is_file():
            manifest.append({'path': str(path.relative_to(a.case)), 'bytes': path.stat().st_size,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in (a.case.with_suffix('.log'), a.case.with_suffix('.exit.json'),
                 a.case.with_suffix('.device-before.txt')):
        if path.is_file():
            manifest.append({'path': '../' + path.name, 'bytes': path.stat().st_size,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (a.out / (a.case.name + '-manifest.json')).write_text(json.dumps(
        {'root': str(a.case.resolve()), 'files': manifest, 'total_bytes': sum(x['bytes'] for x in manifest)}, indent=2) + '\n')
    print(json.dumps({'case': a.case.name, 'status': result['status'], 'candidate_accepted': result['candidate_accepted'],
                      'files': len(manifest), 'bytes': sum(x['bytes'] for x in manifest)}))


if __name__ == '__main__':
    main()
