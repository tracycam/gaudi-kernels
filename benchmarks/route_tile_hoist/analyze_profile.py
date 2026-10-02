"""Complete physical-node coverage and overlapping engine union per invocation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qkv_postprocess'))
from analyze_chain_profile import analyze


def union(intervals):
    total = 0.
    end = None
    for lo, hi in sorted(intervals):
        if end is None or lo >= end:
            total += hi-lo
            end = hi
        elif hi > end:
            total += hi-end
            end = hi
    return total


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    graphs=[]
    for path in (a.case/'post_graph.json').rglob('*.json'):
        graphs.extend(g for g in json.loads(path.read_text())['graphs']
                      if any(n['guid']=='gk_mxfp4_graph_decode_historical' for n in g['nodes']))
    assert len(graphs)==1
    trace=list((a.case/'trace').glob('mac_accel*.json'))
    assert len(trace)==1
    detailed=analyze(trace[0],a.case/'post_graph.json',[graphs[0]['name']],
                     {'gemm':'GEMM','batch_gemm':'BatchGemm'})
    rows=[]
    for invocation in detailed['invocations']:
        intervals=defaultdict(list)
        for node in invocation['nodes']:
            for pid,tid,lo,hi in node['engine_intervals']:
                intervals[node['op']].append((lo,hi))
        rows.append(dict(recipe_idx=invocation['recipe_idx'],envelope_us=invocation['envelope_us'],
                         physical_nodes=invocation['physical_nodes'],operations=invocation['operations'],
                         operation_union_us={op:union(v)for op,v in intervals.items()},
                         compute_union_us=union([i for v in intervals.values()for i in v])))
    # capture + six validation replays + five warmups + four measured replays.
    assert len(rows)==16, len(rows)
    timed=rows[12:]
    summary=dict(status='PASS_PHYSICAL_NODE_COVERAGE',trace_sha256=detailed['trace_sha256'],
                 rows=rows,timed_indices=[r['recipe_idx']for r in timed],
                 envelope_median_us=statistics.median(r['envelope_us']for r in timed),
                 compute_union_median_us=statistics.median(r['compute_union_us']for r in timed),
                 operation_union_median_us={op:statistics.median(r['operation_union_us'][op]for r in timed)
                                           for op in timed[0]['operation_union_us']},
                 scope='Matched same-run actual PostGraph IDs and native engine B/E events, all physical nodes. '
                       'Last four of sixteen invocations. Per-operation intervals overlap and must not be summed. '
                       'Profile instrumentation excluded from performance ABBA.')
    a.output.with_suffix('.details.json').write_text(json.dumps(detailed)+'\n')
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items()if k!='rows'}))


if __name__=='__main__':main()
