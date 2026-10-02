"""Offline small-M model-integration evidence; no device acquisition."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('--case', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
result = json.loads((a.case/'result.json').read_text())
graphs = json.loads((a.case/'post_graph.json').read_text())['graphs']
timing = []
for t in sorted({r['T'] for r in result['timing']}):
    for state in sorted({r['state'] for r in result['timing']}):
        modes = {}
        for mode in ('production', 'staged', 'expert_stream'):
            rows = [r for r in result['timing'] if (r['T'], r['state'], r['mode']) == (t, state, mode)]
            if rows:
                arms = sorted({r['arm'] for r in rows})
                modes[mode] = statistics.mean(statistics.median(r['event_us'] for r in rows if r['arm'] == arm) for arm in arms)
        timing.append(dict(tokens=t, state=state, event_us=modes,
                           stream_over_production=modes['expert_stream']/modes['production']))
graph_rows = []
for g in graphs:
    tensors = {t['name']: t for t in g['tensors']}
    physical = [n for n in g['nodes'] if not n.get('is_logical')]
    decoders = [n for n in physical if n['guid'] in ('gk_moe_stream_decode_historical_k8', 'gk_mxfp4_graph_decode_historical')]
    expanded = [tensors[o] for n in decoders for o in n['output_tensors']]
    row = dict(graph=g['name'], physical_nodes=len(physical),
               engines=dict(Counter(n['engine'] for n in physical)),
               decoded_output_placement=dict(Counter(t['allocation'] for t in expanded)),
               decoder_tpc_engines=[n.get('tpc_working_engines') for n in decoders],
               guid_counts=dict(Counter(n['guid'] for n in physical)))
    if expanded:
        assert all(t['allocation'] == 'SRAM' and not t['persistent'] for t in expanded)
        row['expanded_bytes_per_invocation'] = sum(2*__import__('math').prod(t['max_shape']) for t in expanded)
    graph_rows.append(row)
geometry = []
for r in result['geometry']:
    active, cap = r['active_experts'], r['capacity']
    geometry.append(dict(**r, original_weight_and_scale_bytes=active*4718592*17//32,
                         bf16_expansion_bytes_including_empty_slots=cap*4718592*2,
                         unused_slots=cap-active,
                         scope='logical operand footprint, not a hardware bus counter'))
summary = dict(status=result['status'], checks=len(result['checks']),
    bad_elements=sum(r.get('bad', 0) for r in result['checks']), timing=timing, graphs=graph_rows,
    geometry=geometry, files_sha256={n:hashlib.sha256((a.case/n).read_bytes()).hexdigest()
                                   for n in ('result.json','post_graph.json','exit.json')},
    scope='Same-input public graph operator comparison; no real-model TPS or observed physical HBM throughput claim')
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(dict(status=summary['status'], checks=summary['checks'], bad_elements=summary['bad_elements'], timing=timing),indent=2))
