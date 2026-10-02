"""Reconstruct useful vs padded route work from sealed inputs and actual PostGraph.

This reports logical arithmetic/requests, not physical HBM counters or measured
peak utilization. The real checkpoint is a single TP-rank/layer witness.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planner import Capacity, assign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.archive/'remote-sha256.json').read_text())
    evidence = {}

    def checked(path):
        name = str(path.relative_to(args.archive))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest['files'][name]['sha256'], name
        evidence[name] = digest
        return path

    # Our generated and SHA-pinned checkpoint witness, not an untrusted pickle.
    state = torch.load(checked(args.archive/'fixtures/inputs-rank0.pt'),
                       map_location='cpu', weights_only=False)
    ids = state['ids']
    assert ids.dtype == torch.int32 and ids.shape == (512, 8)
    counts = Counter(ids.flatten().tolist())
    rows = []
    products_per_route = 6144*512 + 256*6144
    for c in (8, 16, 32):
        cap = Capacity(512, 8, 384, c)
        mapping = assign(ids.tolist(), cap)
        active = sum(n > 0 for n in mapping.tile_valid_rows)
        cost = cap.costs(active, len(counts), n_tile=2048)
        useful = ids.numel()*2*products_per_route
        declared = cap.max_tiles*cap.rows*2*products_per_route
        cost.update(useful_logical_MAC_flops=useful,
                    declared_padded_MAC_flops=declared,
                    useful_fraction_of_declared_MAC=useful/declared,
                    precision='BF16 MME inputs with FP32 results',
                    shape_device_qualified_here=c == 32)
        rows.append(cost)

    graphs = []
    post = args.archive/'results'/args.case/'post_graph.json'
    for path in sorted(post.rglob('*.json')) if post.is_dir() else [post]:
        for graph in json.loads(checked(path).read_text()).get('graphs', []):
            dec = [n for n in graph['nodes']
                   if n['guid'] == 'gk_mxfp4_graph_decode_historical']
            if not dec:
                continue
            tensors = {t['name']: t for t in graph['tensors']}
            nodes = [n for n in graph['nodes'] if n['guid'] in ('gemm', 'batch_gemm')]
            flops = 0
            shapes = Counter()
            estimates = Counter()
            decoded_bytes = 0
            for node in nodes:
                assert node['params'] == [0, 0], 'Only non-transposed compiled GEMMs are covered'
                a, b = [tensors[n]['max_shape'] for n in node['input_tensors']]
                y = tensors[node['output_tensors'][0]]['max_shape']
                assert a[0] == b[1] and y[:2] == [b[0], a[1]]
                assert math.prod(a[2:]) == math.prod(b[2:]) == math.prod(y[2:])
                flops += 2*a[0]*math.prod(y)
                shapes[(tuple(a), tuple(b), tuple(y))] += 1
                estimates[round(node['mme_compute_utilization'], 8)] += 1
            for node in dec:
                tensor = tensors[node['output_tensors'][0]]
                assert tensor['allocation'] == 'SRAM' and not tensor['persistent']
                decoded_bytes += math.prod(tensor['max_shape'])*tensor['dtype_bit_size']//8
            expected = rows[-1]
            assert flops == expected['declared_padded_MAC_flops']
            assert decoded_bytes == expected['all_decoded_store_bytes_including_empty']
            graphs.append(dict(name=graph['name'], MME_nodes=len(nodes), decoder_nodes=len(dec),
                declared_MAC_flops_from_actual_nodes=flops,
                decoded_output_payload_from_actual_nodes=decoded_bytes,
                shapes=[dict(A=k[0], B=k[1], Y=k[2], nodes=v) for k,v in shapes.items()],
                compiler_estimated_compute_utilization_nodes=dict(estimates),
                compiler_estimate_is_not_measured_utilization=True))
    assert graphs
    report = dict(status='PASS_SEALED_CHECKPOINT_LOGICAL_WORK_RECONSTRUCTION',
        T=512, routes=8, experts=384, active_experts=len(counts),
        actual_expert_rows_histogram=dict(Counter(counts.values())), geometries=rows,
        compiled_graphs=graphs, verified_inputs=evidence,
        scope='One real checkpoint TP-rank0 layer witness. C8/C16 are planning alternatives only; '
              'C32 is reconstructed from actual compiled nodes. Logical decoded stores are SRAM payload, '
              'not expanded HBM weights. Compiler estimates and padded arithmetic are not hardware counters. '
              'No standalone or model speedup is inferred from these counts.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], active_experts=len(counts),
        rows=[{k:r[k] for k in ('tile_rows','max_tiles','actual_tiles','empty_tiles',
                              'useful_fraction_of_declared_MAC','requested_original_bytes',
                              'empty_zero_store_bytes')} for r in rows])))


if __name__ == '__main__':
    main()
