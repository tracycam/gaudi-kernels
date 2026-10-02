#!/usr/bin/env python3
"""Union physical engine intervals in a separate Synapse TEF JSONL capture.

Trace timestamps are microseconds. They are not aligned to host monotonic time.
Multiple TPC cores/MME lanes are unioned, never added as wall time.
"""
import argparse
import json
from pathlib import Path


ENGINES = ('TPC', 'MME', 'NIC Internal', 'PDMA', 'EDMA')


def union(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if end < begin:
            raise ValueError('Negative engine interval')
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((begin, end))
    return merged


def duration(intervals):
    return sum(end-begin for begin, end in union(intervals))


def events(path):
    with path.open() as source:
        for line in source:
            yield json.loads(line)


def analyze(path, replays):
    if replays < 1:
        raise ValueError('Positive captured replay count required')
    processes = {}
    # Read metadata first: do not depend on metadata/event ordering in the dump.
    for event in events(path):
        if event.get('ph') == 'M' and event.get('name') == 'process_name':
            name = event.get('args', {}).get('name', '')
            for engine in ENGINES:
                if name.startswith('*'+engine+' ('):
                    processes[event['pid']] = engine
    if set(processes.values()) != set(ENGINES):
        raise ValueError('Missing physical engine metadata')
    intervals = {engine: [] for engine in ENGINES}
    pending = {}
    for event in events(path):
        if event.get('pid') not in processes or event.get('ph') not in ('B', 'E'):
            continue
        key = (event['pid'], event['tid'], event['id'])
        if event['ph'] == 'B':
            if key in pending:
                raise ValueError('Duplicate begin event')
            pending[key] = event['ts']
        else:
            if key not in pending:
                raise ValueError('End without begin event')
            begin = pending.pop(key)
            if event['ts'] < begin:
                raise ValueError('Negative engine interval')
            intervals[processes[event['pid']]].append((begin, event['ts']))
    if pending:
        raise ValueError('Unclosed physical engine events')
    all_intervals = [interval for values in intervals.values() for interval in values]
    if not all_intervals:
        raise ValueError('No physical engine execution captured')
    begin = min(i[0] for i in all_intervals)
    end = max(i[1] for i in all_intervals)
    compute = intervals['TPC']+intervals['MME']
    compute_intervals = union(compute)
    compute_busy = duration(compute)
    selected_busy = duration(all_intervals)
    gaps = [right[0]-left[1] for left, right in zip(compute_intervals, compute_intervals[1:])]
    bins = {}
    for label, lower, upper in (('under_3us', 0, 3), ('3_to_10us', 3, 10),
                                ('10_to_20us', 10, 20), ('20_to_60us', 20, 60),
                                ('at_least_60us', 60, float('inf'))):
        values = [gap for gap in gaps if lower <= gap < upper]
        bins[label] = {'average_count_per_replay': len(values)/replays,
                       'average_total_ms_per_replay': sum(values)/replays/1000}
    return {'source': str(path), 'captured_replays': replays,
            'paired_intervals': {engine: len(values) for engine, values in intervals.items()},
            'compute_gap_bins': bins,
            'compute_edge_idle_ms_per_replay': (end-begin-compute_busy-sum(gaps))/replays/1000,
            'average_per_replay_ms': {
                'window': (end-begin)/replays/1000,
                **{engine: duration(values)/replays/1000 for engine, values in intervals.items()},
                'compute_union': compute_busy/replays/1000,
                'TPC_MME_overlap': (duration(intervals['TPC'])+duration(intervals['MME'])-compute_busy)/replays/1000,
                'selected_engine_union': selected_busy/replays/1000,
                'compute_idle': (end-begin-compute_busy)/replays/1000,
                'selected_engine_idle': (end-begin-selected_busy)/replays/1000},
            'scope': 'Physical TPC/MME/NIC Internal/PDMA/EDMA interval unions. Total capture window divided by supplied replay count, including inter-replay gaps. Engine times overlap and are not additive. Selected-engine idle is not a claim about all hardware engines. Profiler overhead included here; excluded from throughput. Trace clock is not host monotonic.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--replays', required=True, type=int)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = json.dumps(analyze(a.trace, a.replays), indent=2)+'\n'
    if a.output:
        a.output.write_text(result)
    else:
        print(result, end='')
