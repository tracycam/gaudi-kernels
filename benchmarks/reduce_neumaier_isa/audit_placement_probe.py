"""Audit paired compiled recipes; placement failure remains explicit."""
import argparse
import json
import math
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'block_fp8_framework'))
from audit import audit

summary = json.loads((a.directory / 'summary.json').read_text())
graphs = json.loads((a.directory / 'post_graph.json').read_text())['graphs']
rows = []
for graph in graphs:
    nodes = [n for n in graph['nodes'] if not n['is_logical']]
    if not any(n['engine'] == 'MME' for n in nodes):
        continue
    reducers = [n for n in nodes if n['guid'] in ('gk_block128_reduce_neumaier_row1_v1',
                                                 'gk_neumaier_isa_handschedule_v1')]
    assert len(reducers) == 1, graph['name']
    reducer = reducers[0]
    tensors = {t['name']: t for t in graph['tensors']}
    partial = tensors[reducer['input_tensors'][0]]
    shape = partial['max_shape']
    case = {(3392, 1, 48): 'qkv-measured-m1', (128, 16, 1): 'single-group-m16'}[tuple(shape)]
    mode = 'quant_in_graph' if any(n['guid'] == 'gk_block128_quant_v1' for n in nodes) else 'external_quant'
    variant = 'hand' if 'isa' in reducer['guid'] else 'old'
    outputs = [tensors[name] for n in nodes if n['engine'] == 'MME' for name in n['output_tensors']]
    placement = audit(graph)
    rows.append(dict(case=case, mode=mode, variant=variant, graph=graph['name'],
                     partial_logical_bytes=math.prod(shape)*4, partial=partial,
                     reducer_bundle=reducer.get('bundle_index'),
                     mme_outputs_sram=all(t['allocation'] == 'SRAM' and not t['persistent'] for t in outputs),
                     consumer_partial_sram=partial['allocation'] == 'SRAM' and not partial['persistent'],
                     physical_nodes=[{k: n.get(k) for k in ('guid', 'engine', 'bundle_index', 'input_tensors', 'output_tensors')} for n in nodes],
                     placement=placement))
actual = {(r['case'], r['mode'], r['variant']) for r in rows}
expected = {(r['case'], r['mode'], r['variant']) for r in summary['captures']}
complete = len(rows) == len(expected) == 8 and actual == expected
candidate = [r for r in rows if r['mode'] == 'external_quant']
sram = complete and all(r['consumer_partial_sram'] and r['mme_outputs_sram'] for r in candidate)
reads = complete and all(r['placement']['no_logical_weight_read_amplification'] for r in rows)
passed = complete and summary.get('numerics_pass', False) and reads
result = dict(status='DIAGNOSTIC_COMPLETE_SRAM_PASS' if passed and sram else
              'DIAGNOSTIC_COMPLETE_SRAM_FAIL' if passed else 'FAIL_DIAGNOSTIC',
              diagnostic_complete=passed, sram_placement_pass=sram,
              all_requirements_pass=passed and sram, expected_graphs=len(expected), found_graphs=len(rows),
              numerical_gate_pass=summary.get('numerics_pass', False), no_logical_weight_read_amplification=reads,
              timing_measured=False, physical_HBM_measured=False, production_changed=False,
              records=rows)
(a.directory / 'placement-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'records'}))
# A completed diagnostic is allowed to establish a negative placement result.
# Its SRAM/all-requirements acceptance flags remain false; no performance claim.
raise SystemExit(0 if passed else 1)
