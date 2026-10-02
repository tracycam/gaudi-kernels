"""Measure per-core SPU intervals without inventing clock or stall counters."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('trace', type=Path)
p.add_argument('--ops', nargs='+', default=['ub_direct_gp', 'ub_direct_down', 'nm_gemv', 'diag_direct_gp_imm1', 'diag_direct_gp_broadcast','downact_direct_down_broadcast'])
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
events = json.loads(a.trace.read_text())['traceEvents']
names = {(e['pid'], e['tid']): e['args']['name'] for e in events
         if e['ph'] == 'M' and e['name'] == 'thread_name'}
stacks = {}
nodes = defaultdict(lambda: defaultdict(list))
for e in events:
    args = e.get('args', {})
    if args.get('op') not in a.ops:
        continue
    core = (e['pid'], e['tid'])
    node = (args.get('Recipe name'), args['op'], e['name'])
    if e['ph'] == 'B' and args.get('HW event name') == 'TPC_SPU_START':
        assert core not in stacks
        stacks[core] = (node, e['ts'])
    elif e['ph'] == 'E' and args.get('HW event name') == 'TPC_SPU_START_TO_SPU_HALT':
        start_node, start = stacks.pop(core)
        assert start_node == node
        nodes[node][core].append([start, e['ts']])
assert not stacks
result = {'trace_sha256': hashlib.sha256(a.trace.read_bytes()).hexdigest(),
          'clock_and_stall_counters': 'UNMEASURED', 'nodes': []}
for (recipe, op, name), cores in nodes.items():
    counts = {len(v) for v in cores.values()}
    assert len(counts) == 1, (op, counts)
    record = {'recipe': recipe, 'op': op, 'name': name, 'active_cores': len(cores), 'launches': []}
    for repeat in range(next(iter(counts))):
        intervals = {names.get(core, str(core)): values[repeat] for core, values in cores.items()}
        starts, ends = zip(*intervals.values())
        durations = [b-a for a, b in intervals.values()]
        union = max(ends)-min(starts)
        record['launches'].append({'core_intervals_us': intervals, 'union_us': union,
            'start_skew_us': max(starts)-min(starts), 'core_min_us': min(durations),
            'core_median_us': statistics.median(durations), 'core_max_us': max(durations),
            'active_core_fraction': sum(durations)/(len(cores)*union)})
    result['nodes'].append(record)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'nodes': len(result['nodes']), 'output': str(a.output)}))
