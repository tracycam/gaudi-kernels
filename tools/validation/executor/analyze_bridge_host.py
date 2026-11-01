"""Summarize nested host-call observations within steady decode steps."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def analyze(document, skip):
    ranks = []
    for rank in document['host_timings']:
        records = rank['records']
        forwards = [r for r in records if r['name'] == 'runner._execute_model_generic'
                    and r['phase'] == 'decode'][skip:]
        totals = defaultdict(list)
        windows = []
        for forward in forwards:
            parents = [r for r in records if r['name'] == 'runner.sample_tokens'
                       and r['begin_ns'] <= forward['begin_ns'] and
                       r['end_ns'] >= forward['end_ns']]
            if len(parents) != 1:
                raise ValueError('Expected one sample_tokens parent for each decode forward')
            window = parents[0]
            sums = defaultdict(int)
            for row in records:
                if row['begin_ns'] >= window['begin_ns'] and row['end_ns'] <= window['end_ns']:
                    sums[row['name']] += row['duration_ns']
            for name, value in sums.items():
                totals[name].append(value / 1e6)
            windows.append(dict(begin_ns=window['begin_ns'], end_ns=window['end_ns']))
        ranks.append(dict(rank=rank['rank'], decode_steps=len(windows),
            dropped=rank['dropped'], host_inclusive_ms={name:dict(count=len(values),
                median=statistics.median(values), mean=statistics.mean(values),
                minimum=min(values), maximum=max(values)) for name, values in totals.items()}))
    return dict(model_layers=document['model_layers'], skipped_decode_steps=skip, ranks=ranks,
        scope='Inclusive nested HOST wall durations from a separate diagnostic request; '
              'parents include children, copies may include DEVICE waits. '
              'Do not sum rows or label forward submission time as GPU compute time.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('result', type=Path)
    parser.add_argument('--skip', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.skip < 0:
        parser.error('skip must be nonnegative')
    result = analyze(json.loads(args.result.read_text()), args.skip)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for rank in result['ranks']:
        print(json.dumps(dict(rank=rank['rank'], steps=rank['decode_steps'],
            medians={k: v['median'] for k, v in rank['host_inclusive_ms'].items()})))


if __name__ == '__main__':
    main()
