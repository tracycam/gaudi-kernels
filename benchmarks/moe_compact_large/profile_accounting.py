"""Offline exact large-M profile replay attribution, independent of CSV names."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qkv_postprocess'))
from analyze_chain_profile import analyze

p = argparse.ArgumentParser()
p.add_argument('--case', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
gate = json.loads((a.case / 'result.json').read_text())
assert gate['status'] == 'PASS_FULL_MOE_PAIRED' and gate['profile_only']
assert gate['experimental_rows'] == 512
graphs = [g for path in (a.case / 'post_graph.json').rglob('*.json')
          for g in json.loads(path.read_text())['graphs']]
mapping = {}
for g in graphs:
    guids = Counter(n['guid'] for n in g['nodes'] if not n['is_logical'])
    if guids['gk_moe_gp_scale_tail_v1']:
        assert guids['downact_direct_down_broadcast'] == 1
        mapping[g['name']] = 'compact'
    else:
        assert guids['nm_gemv'] == 2 and guids['nm_prep'] == 1
        mapping[g['name']] = 'broadcast'
assert Counter(mapping.values()) == {'compact': 1, 'broadcast': 1}
traces = list((a.case / 'trace').glob('*.json'))
assert len(traces) == 1
result = analyze(traces[0], a.case / 'post_graph.json', list(mapping))
ordered = sorted(result['invocations'], key=lambda r: r['start_us'])
expected = [sample for sample in gate['timing'] for _ in range(sample['replays'])]
assert len(expected) == 36
prefix, timed = ordered[:-len(expected)], ordered[-len(expected):]
assert Counter(row['recipe'] for row in prefix) == Counter({name: 18 for name in mapping})
summary = {}
for row, sample in zip(timed, expected):
    assert mapping[row['recipe']] == sample['variant']
    row.update(case=sample['case'], variant=sample['variant'], trial=sample['trial'])
for case in sorted({x['case'] for x in timed}):
    summary[case] = {}
    for variant in ('broadcast', 'compact'):
        rows = [r for r in timed if r['case'] == case and r['variant'] == variant]
        assert len(rows) == 6
        operations = defaultdict(list)
        cores = defaultdict(set)
        gaps = []
        for row in rows:
            counts = Counter()
            for n in row['nodes']:
                counts[n['op']] += 1
                key = n['op'] + '#' + str(counts[n['op']])
                operations[key].append(n['envelope_us'])
                cores[key].add(n['engine_count'])
            # Nodes in this graph are sequential; verify rather than summing
            # durations through possible overlaps.
            intervals = sorted((n['start_offset_us'], n['end_offset_us']) for n in row['nodes'])
            assert all(x[1] <= y[0] for x, y in zip(intervals, intervals[1:]))
            gaps.append(sum(y[0] - x[1] for x, y in zip(intervals, intervals[1:])))
        assert all(v == {24} for v in cores.values())
        summary[case][variant] = dict(samples=len(rows), physical_nodes=rows[0]['physical_nodes'],
            native_recipe_envelope_us=statistics.median(r['envelope_us'] for r in rows),
            physical_node_envelope_us={k: statistics.median(v) for k, v in operations.items()},
            all_24_cores=True, nonoverlapping_node_gaps_us=statistics.median(gaps))
result.update(summary=summary, timed_invocations=timed,
              source_result_sha256=hashlib.sha256((a.case / 'result.json').read_bytes()).hexdigest(),
              source_case=str(a.case),
              sequence_assignment='36 qualification (18 per recipe), then 36 timed invocations; actual recipe IDs, complete physical nodes and per-engine B/E events checked. No host intervals counted.')
a.output.parent.mkdir(parents=True, exist_ok=True)
assert not a.output.exists()
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(summary, indent=2))
