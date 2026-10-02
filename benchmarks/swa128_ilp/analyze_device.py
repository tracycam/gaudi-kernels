"""Bounded profile-c interpretation; no host-floor subtraction."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'swa128_attention'))
from analyze_profile import analyze
p = argparse.ArgumentParser()
p.add_argument('--gate', type=Path, required=True)
p.add_argument('--profile', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
gate = json.loads((a.gate / 'result.json').read_text())
profile = json.loads((a.profile / 'result.json').read_text())
assert gate['status'] == profile['status'] == 'PASS_FULL_ATTENTION'
assert [(r['position'], r['kind']) for r in profile['records']] == [(128, 'ordinary'), (32767, 'ordinary'), (0, 'ordinary')]
csv_path, = list((a.profile / 'trace').glob('*analyzed_nodes.csv'))
rows = list(csv.DictReader(csv_path.open()))
graphs = analyze(csv_path)['graphs']
result = {'gate_cases': len(gate['records']), 'context_bf16_words_per_variant': len(gate['records']) * 2048,
          'fp32_score_words_compared': len(gate['records']) * 2048,
          'max_relative_l2_fp64': max(r['variants']['quad']['relative_l2_fp64'] for r in gate['records']),
          'csv_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(), 'native': {},
          'scope': 'complete three-node producer-attention-consumer graph; explicit TPC body and engine envelope, no model TPS'}
for variant, count in [('baseline', 20), ('quad', 17)]:
    nodes = sorted((r for r in rows if r['Op Type'] == f'gk_swa128_ilp_{variant}_v0'),
                   key=lambda r: float(r['Start time of node']))
    assert len(nodes) == count and all(r['Parallel Engines'] == '16' for r in nodes)
    graph, = [g for g in graphs.values() if f'gk_swa128_ilp_{variant}_v0' in g['ops_per_invocation']]
    assert graph['nodes_per_invocation'] == 3 and graph['invocation_count'] == count
    def group(first, end):
        chosen = graph['invocations'][first:end]
        return {'ordinals': list(range(first, end)),
                'attention_tpc_median_us': statistics.median(float(r['Duration (us)']) for r in nodes[first:end]),
                'chain_envelope_median_us': statistics.median(r['envelope_us'] for r in chosen),
                'chain_envelope_us': [r['envelope_us'] for r in chosen]}
    # Capture execution + one warm replay precede each graph's dynamic cases.
    # Each complete case is one correctness replay + six ABBA replays/variant.
    result['native'][variant] = {'physical_nodes': 3, 'parallel_tpc_engines': 16,
                                 'position128': group(2, 9), 'position32767': group(9, 16),
                                 'cold_partial_trace': group(16, count),
                                 'cold_comparison_qualified': False}
assert result['native']['quad']['cold_partial_trace']['ordinals'] == [16]
result['unprofiled_abba'] = gate['abba_sequence']
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'unprofiled_abba'}, indent=2))
