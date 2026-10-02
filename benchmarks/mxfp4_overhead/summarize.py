"""Summarize retained logs without converting numerical failures to success."""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def summarize(root):
    runs = []
    for directory in sorted(root.iterdir()):
        if not (directory / 'run.log').exists():
            continue
        records = []
        for line in (directory / 'run.log').read_text(errors='replace').splitlines():
            if not line.startswith('{'):
                continue
            # C printf writes inf; preserve raw logs and normalize only JSON output.
            line = re.sub(r'(?<=:)inf(?=[,}])', 'Infinity', line)
            line = re.sub(r'(?<=:)-inf(?=[,}])', '-Infinity', line)
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
        row = {'case': directory.name}
        for stage in ('prepared', 'plan', 'correctness'):
            row[stage] = next((r for r in records if r.get('stage') == stage), None)
        row['timings'] = [r for r in records if r.get('stage') == 'timing']
        for name in ('event_us', 'wall_us'):
            values = [r[name] for r in row['timings']]
            row['median_' + name] = statistics.median(values) if values else None
        for name in ('launch', 'exit'):
            path = directory / (name + '.json')
            row[name] = json.loads(path.read_text()) if path.exists() else None
        row['passed'] = bool(row['exit'] and row['exit'].get('returncode') == 0
                             and row['correctness'] and row['correctness']['checked'] > 0
                             and row['correctness']['bad'] == 0)
        row['profile'] = []
        for path in directory.glob('trace/*_analyzed_nodes.csv'):
            for node in csv.DictReader(path.open()):
                row['profile'].append({key: node[key] for key in (
                    'Node Name', 'Duration (us)', 'Avg single engine (us)',
                    'Start time of node', 'End time of node', 'Input placement',
                    'Output placement', 'Num activations', 'Parallel Engines')})
        runs.append(row)
    return finite_json({
        'all_pass': all(row['passed'] for row in runs),
        'qualification': {
            'status': 'experimental_opt_in_not_fully_qualified',
            'default_n256_changed': False,
            'strict_no_extra_weight_bytes_for_arbitrary_tails': False,
            'tail_padding_example': {'N': 513, 'K': 257, 'checkpoint_bytes': 70794, 'prepared_bytes': 156672},
            'accepted_e8m0_codes': [2, 252],
            'unsupported': 'FP32 subnormal products/results do not have full support; retained final strict-gate failure',
            'final_failed_case': 'exact512-subnormal-mixed-v8',
            'historical_comparator_activation_nonzero_domain': '[2^-118, 2^119]',
        },
        'runs': runs,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = summarize(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'runs': len(result['runs']), 'all_pass': result['all_pass']}))
