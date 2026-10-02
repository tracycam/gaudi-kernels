#!/usr/bin/env python3
"""Read only sealed public PostGraphs and production traces; no runtime/device imports."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.canonical
    remote = root / 'artifacts/builds/reduce-neumaier-isa-public/device-f1fe27e-c94fddc/remote'
    post = remote / 'results/neumaier-hand-public-all-a/public/post_graph.json'
    production = root / 'artifacts/builds/production-integration/model-runs/production-isa-70-b'
    result = {'device_accessed': False, 'postgraph': str(post.relative_to(root)),
              'postgraph_sha256': digest(post), 'graphs': []}
    for g in json.loads(post.read_text())['graphs']:
        nodes = [n for n in g['nodes'] if not n['is_logical']]
        if not any(n.get('engine') == 'MME' for n in nodes):
            continue
        tensors = {t['name']: t for t in g['tensors']}
        reduce = next(n for n in nodes if 'neumaier' in n.get('guid', ''))
        partial = tensors[reduce['input_tensors'][0]]
        item = {'name': g['name'], 'workspace_bytes': g['workspace_size'],
                'partial': partial, 'partial_logical_bytes': math.prod(partial['max_shape']) * 4,
                'reducer_guid': reduce['guid'], 'reducer_bundle': reduce.get('bundle_index'),
                'physical_nodes': []}
        for n in nodes:
            row = {k: n.get(k) for k in ('name', 'engine', 'guid', 'bundle_index', 'exec_order_idx')}
            for key in ('input_tensors', 'output_tensors'):
                row[key] = [{k: tensors[name].get(k) for k in
                             ('name', 'max_shape', 'allocation', 'offset', 'alias', 'persistent', 'in_persist_storage')}
                            for name in n[key]]
            item['physical_nodes'].append(row)
        result['graphs'].append(item)
    libraries = []
    for name in ('libblock_fp8_reduce_experiment.so', 'block_fp8_reduce_experiment.so',
                 'libgaudi_block_fp8_framework_tpc.so', 'gaudi_block_fp8_torch.so'):
        a = remote / 'runtime' / name
        b = production / 'source/production-runtime/libraries' / name
        libraries.append({'name': name, 'standalone_sha256': digest(a),
                          'production_sha256': digest(b), 'identical': digest(a) == digest(b)})
    result['frozen_libraries'] = libraries
    trace = production / 'native-device-profile-rank0.jsonl'
    counts = collections.Counter()
    names = collections.defaultdict(set)
    examples = {}
    for line in trace.open():
        event = json.loads(line)
        op = event.get('args', {}).get('op', '')
        if op not in ('gk_block128_quant_v1', 'BatchGemm', 'gk_block128_reduce_neumaier_row1_v1'):
            continue
        name = event.get('name', '')
        category = op + ('/bundle' if '_bundle_' in name else '/unbundled')
        counts[category] += 1
        names[category].add(name)
        examples.setdefault(category, event)
    result['production_trace'] = {'path': str(trace.relative_to(root)), 'sha256': digest(trace),
        'event_counts_not_kernel_launch_counts': dict(counts),
        'unique_node_name_counts': {k: len(v) for k, v in names.items()}, 'examples': examples,
        'tensor_placement_available': False,
        'graph_compiler_log_bytes': (production / 'habana-logs/graph_compiler.log').stat().st_size,
        'postgraph_files': [str(p.relative_to(production)) for p in production.rglob('*post*graph*.json')]}
    result['limits'] = ['Production trace is topology evidence, not tensor location or physical HBM counters.',
        'Exact proprietary compiler bundle-rejection reason is not present in the preserved logs.',
        'SRAM gate remains FAIL; no graph or production implementation changed.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'graphs': len(result['graphs']), 'libraries_identical': all(x['identical'] for x in libraries),
                      'partial_bytes': [x['partial_logical_bytes'] for x in result['graphs']],
                      'production_categories': dict(counts)}))


if __name__ == '__main__':
    main()
