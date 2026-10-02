"""Attribute completed native trace intervals to actual recipe families.

TPC/MME intervals include internal stalls; this is neither useful ALU time nor
a removable-overhead estimate. No device access or reclassification of a run.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case', type=Path)
    p.add_argument('--executor', type=Path, required=True)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--position', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--result', type=Path, help='Explicit immutable progress snapshot when the full run is ongoing')
    a = p.parse_args()
    sys.path.insert(0, str(a.executor / 'executor'))
    from analyze_profile import read, duration

    result = json.loads((a.result or a.case / 'result.json').read_text())
    rank = result['device_profile']['ranks'][a.rank]
    profile = rank['device_profile']
    assert not profile['start_code'] and not profile['stop_code'] and not profile['remaining']
    index = profile['positions'].index(a.position)
    launches = rank['captures'][0]['plan_info'][3]
    trace = a.case / f'native-device-profile-rank{a.rank}.jsonl'
    events = [json.loads(line) for line in trace.open()]
    starts = sorted((e for e in events if e['ph'] == 'V' and e['cat'] == 'Recipe Start'), key=lambda e: e['ts'])
    assert len(starts) == launches * len(profile['positions'])
    times = [e['ts'] for e in starts]
    audit, rows = read(trace, require_gate=False)
    assert not audit['errors'], audit['errors']
    by_launch = defaultdict(list)
    for row in rows:
        if (row['engine'] not in ('TPC', 'MME') or row['end_us'] <= row['start_us']
                or row['op'] == 'null' or row['name'] in ('', 'null', 'null descriptor')
                or 'write to mem' in row['name']):
            continue
        launch = bisect_right(times, row['start_us'] + .001) - 1
        if not index * launches <= launch < (index + 1) * launches:
            continue
        assert row['end_us'] <= starts[launch]['ts'] + starts[launch]['dur'] + .01
        if launch + 1 < len(starts):
            assert row['end_us'] <= times[launch + 1] + .01
        by_launch[launch].append(row)
    families = defaultdict(list)
    for launch, members in by_launch.items():
        ops = {r['op'] for r in members}
        kind = ('moe' if ops & {'gk_moe_gp_scale_tail_v1', 'gk_moe_gp_folded_v1',
                               'gk_moe_stream_decode_historical_k8'} else
                'swa_qkv_attention' if any(op and op.startswith('gk_swa128_') for op in ops) else
                'full_qkv_attention' if 'gk_block128_quant_v1' in ops else 'other')
        span = max(r['end_us'] for r in members) - min(r['start_us'] for r in members)
        busy = duration((r['start_us'], r['end_us']) for r in members)
        counts = Counter({op: len({r['name'] for r in members if r['op'] == op}) for op in ops})
        families[kind].append(dict(launch=launch % launches, span_us=span, compute_us=busy,
            internal_gap_us=span-busy, physical_named_nodes=len({(r['engine'], r['name']) for r in members}), ops=dict(counts),
            op_union_us={op:duration((r['start_us'],r['end_us']) for r in members if r['op']==op) for op in ops}))
    totals = {kind: dict(recipes=len(members), **{field: sum(r[field] for r in members)
              for field in ('span_us', 'compute_us', 'internal_gap_us', 'physical_named_nodes')})
              for kind, members in families.items()}
    output = dict(case=a.case.name, rank=a.rank, position=a.position, totals=totals, families=dict(families),
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        scope='one completed diagnostic token; disjoint recipe families; traced engine intervals include stalls; not TPS or removable overhead')
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
