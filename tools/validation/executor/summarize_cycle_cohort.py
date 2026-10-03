"""Summarize synchronized TP cycle diagnostics using the slowest rank per step.

No service TPS claim: these bounded diagnostics use private KV and ignore EOS.
Capture/warmup/profiled cycles are excluded from timing, not from correctness.
"""
import argparse
import json
from pathlib import Path
import statistics


def summarize(directory, ranks=8):
    reports = [json.loads((directory / f'rank{rank}.json').read_text()) for rank in range(ranks)]
    if any(r['rank'] != i or not r['pass'] or r['release_code'] != 0 for i, r in enumerate(reports)):
        raise ValueError('Failed rank or device release')
    if any(len(r['cycles']) != len(reports[0]['cycles']) for r in reports):
        raise ValueError('Unequal cycle counts')
    steps = []
    witness_fields = ('starts', 'emitted', 'proposed_ids', 'target_next_ids', 'emitted_ids', 'matched_drafts')
    for index, reference in enumerate(reports[0]['cycles']):
        cohort = [r['cycles'][index] for r in reports]
        if any(s['step'] != reference['step'] or not s['pass'] for s in cohort):
            raise ValueError('Unequal steps or failed functional check')
        if any(s[field] != reference[field] for s in cohort for field in witness_fields):
            raise ValueError('TP output witness disagreement')
        timed = all(s['execution']['kind'] == 'recorded-replay' and not s['execution']['profiled'] for s in cohort)
        row = {'step': reference['step'], 'timed': timed, 'emitted': reference['emitted']}
        if timed:
            for output, source in (('cycle_wall_ms', 'eager_cycle_wall_ns'),):
                values = [s[source] / 1e6 for s in cohort]
                row[output] = values
                row['max_rank_' + output] = max(values)
            values = [s['execution']['sdk_replay_with_completion_wall_ns'] / 1e6 for s in cohort]
            row['sdk_completion_ms'] = values
            row['max_rank_sdk_completion_ms'] = max(values)
        steps.append(row)
    timed = [s for s in steps if s['timed']]
    if not timed:
        raise ValueError('No unprofiled replay cycles')
    return {'all_rank_witnesses_equal': True, 'ranks': ranks, 'steps': steps,
            'median_max_rank_cycle_ms': statistics.median(s['max_rank_cycle_wall_ms'] for s in timed),
            'median_max_rank_sdk_completion_ms': statistics.median(s['max_rank_sdk_completion_ms'] for s in timed),
            'scope': 'Bounded private-KV functional diagnostics. Rank-local host clocks are durations, not aligned timestamps. Maximum rank duration per synchronized step is conservative cohort accounting, not HTTP throughput or completed-answer quality.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--ranks', type=int, default=8)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = json.dumps(summarize(a.directory, a.ranks), indent=2) + '\n'
    if a.output:
        a.output.write_text(result)
    else:
        print(result, end='')
