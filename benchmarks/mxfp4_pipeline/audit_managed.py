"""Check real compiler-managed N slices and original-byte ownership.

This deliberately accepts only the full-K, N256 layout used by this probe.
Unexpected K slicing, copies, gaps or rereads fail instead of receiving a 1x
label. It is an element-request/placement audit, not an HBM bus counter.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def audit(case):
    logs = [json.loads(s) for s in (case / 'run.log').read_text().splitlines()
            if s.startswith('{')]
    plan = next(r for r in logs if r.get('stage') == 'plan')
    check = next(r for r in logs if r.get('stage') == 'correctness')
    assert check['checked'] > 0 and check['bad'] == 0
    outcome = json.loads((case / 'exit.json').read_text())
    assert outcome['returncode'] == outcome['runner_exit_code'] == 0
    graph_path = case / 'post_graph.json'
    if graph_path.is_dir():
        paths = list(graph_path.glob('*.post.json'))
        assert len(paths) == 1
        graph_path = paths[0]
    graphs = json.loads(graph_path.read_text())['graphs']
    assert len(graphs) == 1
    graph = graphs[0]
    tensors = {t['name']: t for t in graph['tensors']}
    owners = {name: t for name, t in tensors.items()
              if re.fullmatch(r'e\d+_(packed|e8m0)', name)}
    assert owners
    for t in owners.values():
        assert t['persistent'] and t['dtype'] == 'uint8' and t['allocation'] == 'DRAM'
    decoders = [n for n in graph['nodes'] if n['guid'] == 'gk_mxfp4_decode_bf16_v1']
    mmes = [n for n in graph['nodes'] if n['engine'] == 'MME']
    assert decoders and mmes
    intervals = collections.defaultdict(list)
    decoded = set()
    sram = []
    for node in decoders:
        out = tensors[node['output_tensors'][0]]
        n, k = out['max_shape']
        assert k == plan['K'], 'K-sliced decoder needs separate access proof'
        assert out['dtype'] == 'bf16' and out['allocation'] == 'SRAM' and not out['persistent']
        assert out['strides'][:2] == [2, 2*n]
        decoded.add(out['name'])
        sram.append((out['offset'], out['offset'] + 2*n*k))
        block_offset = int.from_bytes(bytes(node['params']), 'little', signed=True)
        assert block_offset >= 0
        for name, width, extent in zip(node['input_tensors'][:2], [128, 256], [k, (k+31)//32]):
            view = tensors[name]
            assert view['allocation'] == 'DRAM' and view['dtype'] == 'uint8'
            assert view['max_shape'][:2] == [width, extent]
            assert view['strides'][:3] == [1, width, width*extent]
            matches = [(on, t) for on, t in owners.items()
                       if t['user_mem_section_index'] == view['user_mem_section_index']]
            assert len(matches) == 1
            owner_name, owner = matches[0]
            start = view['offset'] - owner['offset'] + block_offset*view['strides'][2]
            end = start + ((n+255)//256)*view['strides'][2]
            assert 0 <= start < end <= math.prod(owner['max_shape'])
            intervals[owner_name].append([start, end])
    assert set(intervals) == set(owners)
    for name, regions in intervals.items():
        cursor = 0
        for start, end in sorted(regions):
            assert start == cursor, (name, 'gap or repeated original bytes', regions)
            cursor = end
        assert cursor == math.prod(owners[name]['max_shape'])
    consumed = set()
    for node in graph['nodes']:
        related = set(node['input_tensors']) & decoded
        if not related:
            continue
        assert node['engine'] == 'MME', 'unexpected expanded-weight consumer/copy'
        assert node['input_tensors'][1] in related
        assert tensors[node['input_tensors'][0]]['dtype'] == 'bf16'
        assert tensors[node['output_tensors'][0]]['dtype'] == 'float32'
        consumed |= related
    assert consumed == decoded
    assert all(n['input_tensors'][1] in decoded for n in mmes)
    merged = []
    for lo, hi in sorted(sram):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    times = [r for r in logs if r.get('stage') == 'timing']
    assert len(times) == 5
    assert all(math.isfinite(r[k]) and r[k] > 0 for r in times for k in ('event_us','wall_us'))
    event = statistics.median(r['event_us'] for r in times)
    weight_bytes = sum(math.prod(t['max_shape']) for t in owners.values())
    assert weight_bytes == plan['logical_weight_bytes']
    active_experts = sum(m > 0 for m in map(int, plan['M'].split(',')))
    checkpoint_bytes = active_experts*plan['N']*((plan['K']+1)//2+(plan['K']+31)//32)
    flops = 2*sum(map(int, plan['M'].split(',')))*plan['N']*plan['K']
    return dict(case=case.name, pass_=True, plan=plan, correctness=check,
                profiled='--profile' in outcome['command'],
                decode_nodes=len(decoders), mme_nodes=len(mmes),
                logical_original_read_bytes=weight_bytes,
                checkpoint_packed_and_scale_bytes=checkpoint_bytes,
                prepared_minus_checkpoint_bytes=weight_bytes-checkpoint_bytes,
                no_storage_padding_amplification=weight_bytes <= checkpoint_bytes,
                original_read_intervals=intervals,
                all_expanded_weights_sram=True,
                decoded_sram_address_union_bytes=sum(hi-lo for lo, hi in merged),
                event_median_us=event,
                wall_median_us=statistics.median(r['wall_us'] for r in times),
                effective_original_TBps=weight_bytes/event/1e6,
                useful_TFLOPS=flops/event/1e6, samples=times,
                graph_sha256=hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                scope='Static full-K prepared-owner read intervals; padding reported separately. No physical byte counter, runtime routing or model claim')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    records = []
    for case in sorted(a.results.iterdir()):
        if not (case/'exit.json').exists():
            continue
        try:
            records.append(audit(case))
        except (AssertionError, KeyError, ValueError, StopIteration, OSError) as e:
            records.append(dict(case=case.name, pass_=False, reason=repr(e)))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(records=records), indent=2)+'\n')
    print(json.dumps([dict(case=r['case'], pass_=r['pass_'],
                          event_us=r.get('event_median_us'), reason=r.get('reason'))
                      for r in records]))
