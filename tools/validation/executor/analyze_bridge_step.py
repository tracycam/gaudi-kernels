"""Summarize complete-step ABBA without adding overlapping host durations."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(result):
    rows = []
    variable = next((name for name in ('input_layout_optimized', 'compiled_regions', 'input_optimized', 'event_batching')
                     if any(run.get(name) is not None for run in result['runs'])), 'step_graph')
    for batch in sorted({run['batch'] for run in result['runs']}):
        arms = [run for run in result['runs'] if run['batch'] == batch]
        if all(run.get(variable) is None for run in arms):
            steps = sum(run['resident_step_count'] for run in arms)
            wall = sum(run['resident_continuous_ms'] for run in arms)
            rows.append(dict(batch=batch, configured_ms=wall/steps,
                configured_aggregate_tps=batch*1000*steps/wall,
                per_run_tps=[run['resident_continuous_tps'] for run in arms],
                token_equal=all(run['outputs']==arms[0]['outputs'] for run in arms)))
            continue
        if [run[variable] for run in arms] != [False, True, True, False]:
            raise ValueError('Expected complete ABBA for each batch')
        grouped = {}
        for enabled in (False, True):
            group = [run for run in arms if run[variable] == enabled]
            # Include time BETWEEN engine.step calls, not just inside them.
            steps = sum(run['resident_step_count'] for run in group)
            wall = sum(run['resident_continuous_ms'] for run in group)
            grouped[enabled] = wall / steps
        rows.append(dict(batch=batch, forward_ms=grouped[False], step_ms=grouped[True],
            saved_ms=grouped[False]-grouped[True],
            forward_aggregate_tps=batch*1000/grouped[False],
            step_aggregate_tps=batch*1000/grouped[True],
            token_equal=all(run['outputs'] == arms[0]['outputs'] for run in arms)))
        if variable != 'step_graph':
            for old,new in [('forward_ms','control_ms'),('step_ms','candidate_ms'),
                            ('forward_aggregate_tps','control_aggregate_tps'),('step_aggregate_tps','candidate_aggregate_tps')]:
                rows[-1][new] = rows[-1].pop(old)
    hosts = []
    for observation in result.get('host_timings', []):
        rank = observation['ranks'][0]
        records = rank['records']
        decodes = [r for r in records if r['name'] == 'runner._execute_model_generic'
                   and r['phase'] == 'decode'][-32:]
        parents = [r for r in records if r['name'] == 'runner.sample_tokens' and
                   any(r['begin_ns'] <= d['begin_ns'] and d['end_ns'] <= r['end_ns']
                       for d in decodes)]
        families = {}
        for record in records:
            if any(p['begin_ns'] <= record['begin_ns'] and record['end_ns'] <= p['end_ns']
                   for p in parents):
                families.setdefault(record['name'], []).append(record['duration_ns']/1e6)
        hosts.append(dict(step_graph=observation['step_graph'], event_batching=observation.get('event_batching'),
            input_optimized=observation.get('input_optimized'), rank=rank['rank'],
            input_layout_optimized=observation.get('input_layout_optimized'),
            compiled_regions=observation.get('compiled_regions'),
            sample_steps=len(parents), calls={name: dict(count=len(values),
                median_ms=statistics.median(values), total_ms=sum(values))
                for name, values in families.items()}))
    coverage = []
    for rank in result['ranks']:
        coverage.append(dict(rank=rank['rank'], calls=rank['step_graph_calls'],
            external_executors=rank['external_executors'], plans=[dict(
                **p['stats'], compute=sum(bool(c['compute']) for c in p['commands']))
                for p in rank['step_plans']]))
    return dict(status=result['status'], variable=variable, layers=result['model_layers'], batches=rows,
                coverage=coverage, host_observations=hosts,
                quality=result.get('step_quality'),
                host_scope='Nested inclusive wall times; never sum parent and child. '
                           'CPU return can include device waits.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.result.read_text()))
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['batches'], indent=2))


if __name__ == '__main__':
    main()
