"""Audit allocation and compare full-recipe times, retaining every sample."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('results', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
records = []
for case in sorted(a.results.iterdir()):
    if not case.is_dir() or not (case / 'launch.json').exists():
        continue
    launch = json.loads((case / 'launch.json').read_text())
    outcome = json.loads((case / 'exit.json').read_text()) if (case / 'exit.json').exists() else {}
    if outcome.get('returncode') != 0 or outcome.get('runner_exit_code') != 0:
        records.append({'case': case.name, 'pass': False, 'launch': launch, 'outcome': outcome})
        continue
    command = launch['command']
    buffers = int(command[command.index('--buffers') + 1])
    rows = [json.loads(line) for line in (case / 'run.log').read_text().splitlines()
            if line.startswith('{')]
    plan = next(row for row in rows if row.get('stage') == 'plan')
    correctness = next(row for row in rows if row.get('stage') == 'correctness')
    timings = [row for row in rows if row.get('stage') == 'timing']
    assert correctness['checked'] > 0 and correctness['bad'] == 0
    assert len(timings) == 5
    assert all(math.isfinite(row[key]) and row[key] > 0
               for row in timings for key in ('event_us', 'wall_us'))
    placement = None
    if (case / 'post_graph.json').is_file():
        # Reuse the full logical read coverage and no-expanded-HBM audit.
        audit = case / 'placement-audit.json'
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] /
                        'mxfp4_linear/audit.py'), str(case), '--output', str(audit)],
                       check=True, stdout=subprocess.DEVNULL)
        placement = json.loads(audit.read_text())
        intervals = {(entry['address'], entry['address'] + entry['bytes'])
                     for entry in placement['scratch']}
        starts = sorted({start for start, _ in intervals})
        assert len(starts) <= buffers
        # All aliases within a slot have the same base. Different slots must
        # not overlap, including the differently sized final N tile.
        for left, right in zip(starts, starts[1:]):
            assert max(end for start, end in intervals if start == left) <= right
        placement['unique_scratch_addresses'] = len(starts)
    median = statistics.median(row['event_us'] for row in timings)
    records.append({'case': case.name, 'pass': True, 'buffers': buffers,
                    'profiled': '--profile' in command, 'plan': plan,
                    'correctness': correctness, 'samples': timings,
                    'event_median_us': median,
                    'wall_median_us': statistics.median(row['wall_us'] for row in timings),
                    'effective_weight_TBps': plan['logical_weight_bytes'] / median / 1e6,
                    'placement': placement})
pairs = []
by_name = {row['case']: row for row in records}
for row in records:
    if not row['pass'] or row.get('buffers') != 1 or row.get('profiled'):
        continue
    peer = by_name.get(row['case'].replace('-b1-', '-b2-'))
    if not peer or not peer['pass']:
        continue
    first = a.results / row['case']
    second = a.results / peer['case']
    # Same full inputs and unchanged outputs prove this scheduling experiment
    # does not change the arithmetic or fixture. Never sample a prefix.
    files = sorted(first.glob('*.bin'))
    assert files
    for path in files:
        assert path.read_bytes() == (second / path.name).read_bytes(), path.name
    pairs.append({'baseline': row['case'], 'candidate': peer['case'],
                  'baseline_us': row['event_median_us'],
                  'candidate_us': peer['event_median_us'],
                  'time_saved_percent': 100 * (1 - peer['event_median_us'] / row['event_median_us']),
                  'all_inputs_and_outputs_bitwise_equal': True})
result = {'scope': 'operator recipes, static access audit; no physical HBM counter or model claim',
          'records': records, 'pairs': pairs}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(pairs, indent=2))
