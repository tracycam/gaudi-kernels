#!/usr/bin/env python3
"""Union physical engine intervals in a separate Synapse TEF JSONL capture.

Trace timestamps are microseconds. They are not aligned to host monotonic time.
Multiple TPC cores/MME lanes are unioned, never added as wall time.
"""
import argparse
from collections import deque
import json
import re
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


def chrome_windows(document, specification):
    """Translate reference trace windows without converting epoch times to float."""
    base = specification['base_time_ns']
    target = document['baseTimeNanoseconds']
    if type(base) is not int or type(target) is not int:
        raise ValueError('Integer trace base timestamps required')
    shift = (base-target)/1000
    return [(begin+shift, end+shift) for begin, end in specification['intervals_us']]


def analyze_chrome(document, windows=None):
    """Union exported device lanes inside explicit CPU token windows, if supplied."""
    rows = document['traceEvents']
    device_pids = {r['pid'] for r in rows if r.get('ph') == 'M' and
                   r.get('name') == 'process_name' and
                   r.get('args', {}).get('name', '').startswith('/device:HPU:')}
    lanes = {}
    for row in rows:
        if row.get('pid') not in device_pids or row.get('name') != 'thread_name':
            continue
        name = row.get('args', {}).get('name', '')
        engine = None
        if re.fullmatch(r'(\[D\d+\] )?TPC( \d+)?', name):
            engine = 'TPC'
        elif re.fullmatch(r'(\[D\d+\] )?MME', name):
            engine = 'MME'
        elif name.startswith('[D') and 'Internal DMA' in name:
            engine = 'EDMA'
        elif name == 'PDMA' or re.fullmatch(r'DMA \d+', name):
            engine = 'PDMA'
        elif re.fullmatch(r'Port \d+', name) or name in ('NIC Internal', 'NIC External'):
            engine = 'NIC'
        if engine:
            lanes[(row['pid'], row['tid'])] = engine
    intervals = {name: [] for name in ('TPC', 'MME', 'NIC', 'PDMA', 'EDMA')}
    for row in rows:
        engine = lanes.get((row.get('pid'), row.get('tid')))
        if engine and row.get('ph') == 'X':
            intervals[engine].append((row['ts'], row['ts']+row.get('dur', 0)))
    observed = [span for values in intervals.values() for span in values]
    if not observed:
        raise ValueError('No exported physical engine activities')
    if windows is None:
        windows = [(min(a for a, b in observed), max(b for a, b in observed))]
    windows = sorted(windows)
    if not windows or any(b <= a for a, b in windows) or any(
            left[1] > right[0] for left, right in zip(windows, windows[1:])):
        raise ValueError('Positive, nonoverlapping windows required')
    summaries = []
    for begin, end in windows:
        clipped = {name: [(max(a, begin), min(b, end)) for a, b in values
                          if a < end and b > begin] for name, values in intervals.items()}
        compute = clipped['TPC']+clipped['MME']
        selected = [span for values in clipped.values() for span in values]
        summaries.append(dict(begin_us=begin, end_us=end, window_ms=(end-begin)/1000,
            **{name+'_ms':duration(values)/1000 for name, values in clipped.items()},
            compute_union_ms=duration(compute)/1000,
            selected_engine_union_ms=duration(selected)/1000,
            compute_idle_ms=((end-begin)-duration(compute))/1000,
            selected_engine_idle_ms=((end-begin)-duration(selected))/1000))
    return dict(windows=summaries,
        paired_intervals={name:len(values) for name, values in intervals.items()},
        engines_without_execution_events=[name for name, values in intervals.items() if not values],
        scope='Exported Chrome HPU activity unions clipped to supplied windows; CPU spans excluded. '
              'NIC includes named internal/external ports. Missing execution does not prove idle. '
              'Profiler overhead included; nested engine times overlap and are not additive.')


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
    operations = {}
    pending = {}
    for event in events(path):
        if event.get('pid') not in processes or event.get('ph') not in ('B', 'E'):
            continue
        # SP categories can differ between descriptor issue/completion. The
        # physical engine/thread/context/node identifies the queued interval.
        key = (event['pid'], event['tid'], event['id'], event.get('name', ''))
        if event['ph'] == 'B':
            # DMA descriptors from the same node can overlap while reusing
            # contextId. Queue-order completion pairs these repeated spans;
            # their physical union is counted once, never as descriptor sums.
            operation = event.get('args', {}).get('op') or event.get('name', '').split(' apiId=')[0]
            pending.setdefault(key, deque()).append((event['ts'], operation))
        else:
            if key not in pending:
                raise ValueError('End without begin event')
            begin, operation = pending[key].popleft()
            if not pending[key]:
                del pending[key]
            if event['ts'] < begin:
                raise ValueError('Negative engine interval')
            intervals[processes[event['pid']]].append((begin, event['ts']))
            operations.setdefault((processes[event['pid']], operation), []).append((begin, event['ts']))
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
            'engines_without_execution_events': [engine for engine, values in intervals.items() if not values],
            'missing_execution_warning': 'Metadata without execution events does not establish zero engine activity. Idle values refer only to observed engines.',
            'operator_family_unions': [dict(engine=engine, operation=operation, intervals=len(values),
                                           average_union_ms_per_replay=duration(values)/replays/1000)
                for (engine, operation), values in sorted(operations.items(), key=lambda item: -duration(item[1]))],
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
    p.add_argument('--format', choices=('tef', 'chrome'), default='tef')
    p.add_argument('--replays', type=int)
    p.add_argument('--windows', type=Path,
                   help='Chrome window JSON: integer base_time_ns and intervals_us pairs; '
                        'automatically translated to the target trace clock')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.format == 'tef':
        if a.replays is None or a.windows is not None:
            p.error('TEF requires --replays and does not accept --windows')
        data = analyze(a.trace, a.replays)
    else:
        if a.replays is not None:
            p.error('Chrome uses explicit windows, not --replays')
        document = json.loads(a.trace.read_text())
        windows = chrome_windows(document, json.loads(a.windows.read_text())) if a.windows else None
        data = analyze_chrome(document, windows)
        data['source'] = str(a.trace)
        data['reference_windows'] = str(a.windows) if a.windows else None
    result = json.dumps(data, indent=2)+'\n'
    if a.output:
        a.output.write_text(result)
    else:
        print(result, end='')
