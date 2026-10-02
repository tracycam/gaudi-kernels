"""Offline descriptor/placement and retained-output audit of recovered graphs.

Logical original-width coverage is distinct from physical HBM transactions.
The audit neither treats a replay count as a launch count nor estimates clocks.
"""
import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path


def audit_graphs(path):
    records = []
    for graph in json.loads(path.read_text())['graphs']:
        nodes = graph['nodes']
        if not any(n['guid'] == 'gemm' for n in nodes):
            continue
        tensors = {t['name']: t for t in graph['tensors']}

        def origin(tensor):
            seen = set()
            while tensor.get('alias'):
                assert tensor['name'] not in seen
                seen.add(tensor['name'])
                tensor = tensors[tensor['alias_of']]
            return tensor

        roots = {name: t for name, t in tensors.items()
                 if t['persistent'] and not t.get('alias') and t['dtype'] == 'hfloat8'}
        regions = defaultdict(list)
        readers = defaultdict(list)
        for node in nodes:
            if node.get('is_logical'):
                continue
            for name in node['input_tensors']:
                tensor = tensors[name]
                root = origin(tensor)
                if root['name'] not in roots:
                    continue
                assert tensor['dtype_bit_size'] == 8 and tensor['allocation'] == 'DRAM'
                shape, strides = tensor['max_shape'], tensor['strides']
                span, index = 1, 0
                while index < len(shape) and strides[index] == span:
                    span *= shape[index]
                    index += 1
                for indices in itertools.product(*(range(n) for n in shape[index:])):
                    offset = tensor['offset'] - root['offset']
                    offset += sum(i * s for i, s in zip(indices, strides[index:]))
                    regions[root['name']].append((offset, offset + span))
                readers[root['name']].append({'node': node['name'], 'guid': node['guid'],
                                             'input': name, 'shape': shape})
        coverage = []
        for name, root in roots.items():
            cursor, once = 0, True
            for begin, end in sorted(regions[name]):
                once = once and begin == cursor
                cursor = end
            size = math.prod(root['max_shape'])
            once = once and cursor == size
            coverage.append({'root': name, 'shape': root['max_shape'], 'bytes': size,
                             'descriptor_bytes': sum(b-a for a, b in regions[name]),
                             'coverage_once': once, 'readers': readers[name]})

        descendants, seen = [], set()

        def trace_decoded(name):
            if name in seen:
                return
            seen.add(name)
            t = tensors[name]
            descendants.append({'name': name, 'dtype': t['dtype'],
                                'allocation': t['allocation'], 'persistent': t['persistent']})
            for other in tensors.values():
                if other.get('alias_of') == name:
                    trace_decoded(other['name'])
            for node in nodes:
                if name not in node['input_tensors']:
                    continue
                if node['guid'] == 'gemm':
                    assert node['input_tensors'][1] == name
                else:
                    for output in node['output_tensors']:
                        trace_decoded(output)
        for node in nodes:
            if node['guid'] == 'fp8_linear_decode':
                for name in node['output_tensors']:
                    trace_decoded(name)
        decoded_sram = all(t['dtype'] == 'bf16' and t['allocation'] == 'SRAM'
                           and not t['persistent'] for t in descendants)
        physical_names = {name for n in nodes if not n.get('is_logical')
                          for name in n['input_tensors'] + n['output_tensors']}
        transient = [{'name': t['name'], 'dtype': t['dtype'], 'allocation': t['allocation']}
                     for t in tensors.values() if t['name'] in physical_names
                     and not origin(t)['persistent'] and t['dtype'] in ('float32', 'hfloat8')]
        ms = {t['max_shape'][1] for t in tensors.values()
              if t['persistent'] and t['dtype'] == 'bf16' and len(t['max_shape']) == 2}
        assert len(ms) == 1
        row = {'graph': graph['name'], 'M': next(iter(ms)),
               'node_counts': dict(Counter(n['guid'] for n in nodes)),
               'original_fp8_roots': coverage, 'decoded_weight_descendants': descendants,
               'decoded_sram_only': decoded_sram,
               'quant_and_accumulation_transient_tensors': transient,
               'all_quant_and_accumulation_sram': all(t['allocation'] == 'SRAM' for t in transient),
               'physical_hbm_transactions': 'not_measured',
               'launch_or_submission_count': 'not_inferred_from_postgraph'}
        row['pass'] = bool(coverage) and all(c['coverage_once'] for c in coverage) and decoded_sram
        records.append(row)
    assert records
    return records


def audit_bits(directory):
    import torch
    # Only our own retained fixtures: Torch 2.11 wrote protocol 4, which the
    # local 2.9 weights-only unpickler cannot yet read (opcode 149).
    data = torch.load(directory / 'inputs-outputs-oracles.pt', map_location='cpu', weights_only=False)
    result = json.loads((directory / 'result.json').read_text())
    checked, mismatch = 0, 0
    if directory.name.startswith('projections-'):
        for fixture in data.values():
            for policy in ('per_token_fp8', 'bf16'):
                for changed in ('', '_changed'):
                    baseline = fixture[policy + '_separate' + changed].view(torch.int16)
                    for variant in ('combined_views', 'combined_contiguous'):
                        actual = fixture[policy + '_' + variant + changed].view(torch.int16)
                        checked += actual.numel()
                        mismatch += int((actual != baseline).sum())
    elif directory.name.startswith('quantizer-chain'):
        for variant in ('lut1', 'lut4'):
            for name in ('actual', 'changed'):
                for actual, baseline in zip(data[variant][name], data['existing'][name]):
                    checked += actual.numel()
                    mismatch += int((actual.view(torch.int16) != baseline.view(torch.int16)).sum())
    else:
        for fixture in data.values():
            for variant in ('lut1', 'lut4'):
                for name in ('output', 'changed'):
                    actual, baseline = fixture[variant][name], fixture['existing'][name]
                    checked += actual.numel()
                    mismatch += int((actual.view(torch.int16) != baseline.view(torch.int16)).sum())
    return {'status': result['status'], 'compared_bf16_words_including_changed_input': checked,
            'different_bit_patterns': mismatch}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for name in ['projections-independent-d', 'quantizer-graph-main-e',
                 'quantizer-graph-stream-f', 'quantizer-chain-g']:
        directory = args.root / name
        rows = audit_graphs(directory / 'post_graph.json')
        bits = audit_bits(directory)
        for row in rows:
            row['comparison_baseline'] = (name == 'projections-independent-d'
                                           and len(row['original_fp8_roots']) == 3)
        cases.append({'case': name, 'graphs': rows, 'bits': bits,
                      'all_graphs_pass': all(r['pass'] for r in rows),
                      'candidates_pass': all(r['pass'] for r in rows if not r['comparison_baseline'])
                      and bits['different_bit_patterns'] == 0})
    result = {'status': 'PASS_CANDIDATES_WITH_REJECTED_BASELINE'
              if all(c['candidates_pass'] for c in cases) else 'FAIL',
              'all_graphs_original_weight_coverage_once': all(c['all_graphs_pass'] for c in cases),
              'cases': cases}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'graphs': sum(len(c['graphs']) for c in cases),
                      'bits': [c['bits'] for c in cases]}))
    raise SystemExit(result['status'] == 'FAIL')
