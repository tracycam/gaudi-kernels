"""Summarize the synthetic feedback probe without extrapolating model TPS."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from tools.validation.executor.analyze_bridge_device_trace import engine_family, from_sdk_jsonl, union


def summarize(path):
    result = json.loads((path/'result.json').read_text())
    rows = []
    for rank in result.get('ranks', []):
        for steps in (4, 16):
            modes = {}
            for mode in ('per-step-feedback', 'step-queued', 'block'):
                samples = [value/steps for arm in rank['timings']
                           if arm['steps'] == steps and arm['mode'] == mode
                           for value in arm['block_us']]
                if not samples: continue
                modes[mode] = dict(samples=len(samples), median_us_per_step=statistics.median(samples),
                                   mean_us_per_step=statistics.mean(samples), max_us_per_step=max(samples))
            if len(modes)!=3: continue
            baseline = modes['per-step-feedback']['median_us_per_step']
            queued = modes['step-queued']['median_us_per_step']
            block = modes['block']['median_us_per_step']
            rows.append(dict(rank=rank['rank'], steps=steps, modes=modes,
                block_reduction_fraction=1-block/baseline,
                feedback_barrier_saved_us=baseline-queued,
                block_vs_queued_saved_us=queued-block))
    traces = []
    for file in sorted(path.glob('device-rank*.jsonl')):
        document = from_sdk_jsonl(file)
        events = document['traceEvents']
        threads = {(e['pid'], e['tid']): e['args']['name'] for e in events
                   if e['ph'] == 'M' and e['name'] == 'thread_name'}
        kinds = {}
        for family in ('TPC', 'MME', 'EDMA', 'PDMA', 'NIC'):
            spans = [e for e in events if e['ph'] == 'X'
                     and engine_family(threads.get((e['pid'], e['tid']), '')) == family]
            kinds[family] = dict(events=len(spans), contexts=len({e['id'] for e in spans}),
                union_us=sum(b-a for a,b in union((e['ts'],e['ts']+e['dur']) for e in spans)),
                tracks=dict(Counter(threads[(e['pid'],e['tid'])] for e in spans)))
        traces.append(dict(path=str(file), engines=kinds,
                           unmatched_events=len(document['unmatched_sdk_events'])))
    return dict(status=result['status'], latency=rows, traces=traces,
                collectives_per_step=result.get('collectives_per_step',1), modules=result.get('modules'),
                prequeue_window_pass=result.get('prequeue_window_pass'), queued=result.get('queued'),
                native_coverage=[dict(rank=r['rank'],execution=r.get('final_execution'))
                                 for r in result.get('ranks',[])],
                sdk_counts=[dict(rank=r['rank'], counts=r['sdk_counts']) for r in result.get('ranks', [])],
                stopped_pending_witness=result.get('stopped_pending_witness'),
                completed_while_stopped=result.get('paused_witness'),
                pause_to_dma_ms=result.get('pause_to_dma_ms'),
                scope='Synthetic bounded device feedback, not whole-model TPS. '
                      'Engine unions overlap; missing NIC duration does not prove NIC idle.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(summarize(a.run), indent=2)+'\n')


if __name__ == '__main__': main()
