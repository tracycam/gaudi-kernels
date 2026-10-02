"""Attribute node-envelope unions without adding overlapping engine times.

CSV node durations span their first engine start to last engine end. These
are scheduling envelopes, not ALU cycles or an exact measure of useful work.
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import statistics

def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result

def length(intervals):
    return sum(b - a for a, b in merged(intervals))

def analyze(path):
    rows = list(csv.DictReader(path.open()))
    invocations = collections.defaultdict(list)
    for row in rows:
        match = re.search(r' iter (\d+)$', row['Node Name'])
        invocations[int(match[1]) if match else 1].append(row)
    results = []
    for iteration, group in sorted(invocations.items()):
        by_unit = collections.defaultdict(list)
        begin = min(float(row['Start time of node']) for row in group)
        nodes = []
        for row in group:
            start = float(row['Start time of node']) - begin
            end = float(row['End time of node']) - begin
            by_unit[row['Unit']].append((start, end))
            nodes.append({'name': row['Node Name'], 'unit': row['Unit'],
                'start_us': start, 'end_us': end, 'shape': row['Input shapes (elements)'],
                'strategy': row['MME Node Strategy']})
        tpc = length(by_unit['TPC']); mme = length(by_unit['MME'])
        occupied = length(by_unit['TPC'] + by_unit['MME'])
        span = max(node['end_us'] for node in nodes)
        results.append({'iteration': iteration, 'device_envelope_us': span,
            'tpc_union_us': tpc, 'mme_union_us': mme,
            'tpc_mme_overlap_us': tpc + mme - occupied,
            'tpc_only_us': occupied - mme, 'mme_only_us': occupied - tpc,
            'neither_tpc_nor_mme_us': span - occupied,
            'nodes': sorted(nodes, key=lambda node: node['start_us'])})
    numeric = [key for key in results[0] if key.endswith('_us')]
    return {'case': path.parent.parent.name, 'source': str(path),
        'median': {key: statistics.median(row[key] for row in results) for key in numeric},
        'invocations': results}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    records = [analyze(path) for path in sorted(args.case.glob('*/trace/*analyzed_nodes.csv'))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'scope': 'node scheduling envelopes; overlapping intervals merged',
        'records': records}, indent=2) + '\n')
    for row in records: print(row['case'], json.dumps(row['median']))
