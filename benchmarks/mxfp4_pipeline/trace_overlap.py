"""Union physical TPC/MME intervals from the vendor Chrome trace.

These are engine activity intervals, not VPU issue or HBM traffic counters.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def merge(intervals):
    out = []
    for lo, hi in sorted(intervals):
        assert hi >= lo
        if out and lo <= out[-1][1]:
            out[-1][1] = max(hi, out[-1][1])
        else:
            out.append([lo, hi])
    return out


def measure(path):
    events = json.loads(path.read_text())['traceEvents']
    processes = {e['pid']: e['args']['name'] for e in events
                 if e['ph'] == 'M' and e['name'] == 'process_name'}
    pending = collections.defaultdict(list)
    spans = collections.defaultdict(list)
    nodes = collections.defaultdict(list)
    for e in events:
        engine = next((x for x in ('TPC', 'MME')
                       if processes.get(e.get('pid'), '').startswith('*'+x)), None)
        if not engine or e['ph'] not in ('B', 'E'):
            continue
        args = e.get('args', {})
        if args.get('op') not in ('gk_mxfp4_decode_bf16_v1', 'GEMM', 'gemm'):
            continue
        key = (e['pid'], e['tid'], e['name'], args.get('id'),
               args.get('Unique Node ID'), args.get('recipe id'))
        if e['ph'] == 'B':
            pending[key].append(e['ts'])
        else:
            assert pending[key], ('unmatched end', key)
            span = [pending[key].pop(0), e['ts']]
            spans[engine].append(span)
            nodes[(engine, e['name'])].append(span)
    assert all(not v for v in pending.values()), 'unmatched begin'
    assert spans['TPC'] and spans['MME']
    tpc, mme = merge(spans['TPC']), merge(spans['MME'])
    both = merge(tpc + mme)
    duration = lambda xs: sum(hi-lo for lo, hi in xs)
    origin = min(lo for lo, hi in both)
    return dict(path=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                physical_event_pairs={k:len(v) for k,v in spans.items()},
                span_us=max(hi for lo,hi in both)-origin,
                tpc_union_us=duration(tpc), mme_union_us=duration(mme),
                overlap_us=duration(tpc)+duration(mme)-duration(both),
                neither_us=max(hi for lo,hi in both)-origin-duration(both),
                node_envelopes=[dict(engine=k[0],name=k[1],start_us=min(a for a,b in v)-origin,
                                     end_us=max(b for a,b in v)-origin)
                                for k,v in sorted(nodes.items())],
                scope='One separately profiled recipe; unions of physical engine events, no issue/byte counter')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('roots', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    results = [measure(path) for root in a.roots
               for path in sorted(root.glob('profile*/trace/*_7.json'))]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='node_envelopes'} for r in results]))
