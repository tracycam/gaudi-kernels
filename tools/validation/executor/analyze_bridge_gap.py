"""Compare paired full-step windows; nested host observations are not GPU time."""
import argparse
import json
from pathlib import Path
from statistics import median


def summarize(document):
    result = dict(status=document['status'], layers=document['layers'], arms=[], diagnostics=[])
    for row in document['runs']:
        record = {key: row[key] for key in ('fixture', 'arm', 'active', 'tps', 'p50_ms',
                   'window_steps', 'token_match', 'native_used_all_ranks', 'capture_gate_pass')}
        record['single_rpc'] = row.get('single_rpc', row['active'])
        record['rank0_graph_plans'] = [{key: plan[key] for key in ('commands', 'collectives', 'ready')}
                                      for plan in row['framework'][0]['plans']]
        record['rank0_command_composition'] = [dict(
            compute_recipes=sum(bool(c['compute']) for c in plan['commands']),
            collectives=sum(c['collectives'] for c in plan['commands']),
            mixed_commands=sum(bool(c['compute']) and bool(c['collectives']) for c in plan['commands']))
            for plan in row['framework'][0].get('dependency_diagnostics', [])]
        if row['active']:
            calls = row['native'][0]['native_steps']
            record['rank0_native_medians_ms'] = {
                key: median(call[key] for call in calls)/1e6
                for key in ('native_wall_ns', 'enqueue_ns', 'caller_ns', 'worker_wall_ns')
                if calls and all(key in call for call in calls)}
            record['captures'] = row['native'][0]['captures']
        result['arms'].append(record)
    for observation in document.get('diagnostics', []):
        ranks = []
        for rank in observation['host']:
            events = rank['records']
            native_steps = next(r['native_steps'] for r in observation['native'] if r['rank']==rank['rank']) if all(
                'rank' in r for r in observation['native']) else observation['native'][rank['rank']]['native_steps']
            decode_intervals = [(r['begin_ns'], r['end_ns']) for r in events
                                if r['name']=='runner._execute_model_generic' and r.get('phase')=='decode']
            decode_intervals += [(r['caller_begin_ns'], r['caller_end_ns']) for r in native_steps]
            samples = sorted((r for r in events if r['name']=='runner.sample_tokens' and any(
                r['begin_ns']<=begin and end<=r['end_ns'] for begin,end in decode_intervals)),
                key=lambda r:r['begin_ns'])[8:]
            stages = {}
            for name in sorted({event['name'] for event in events}):
                durations = []
                for sample in samples:
                    contained = [r for r in events if r['name']==name and
                                 sample['begin_ns']<=r['begin_ns'] and r['end_ns']<=sample['end_ns']]
                    if contained:
                        durations.append(sum(r['duration_ns'] for r in contained)/1e6)
                if durations:
                    stages[name] = dict(count=len(durations), median=median(durations))
            ranks.append(dict(rank=rank['rank'], sampled_windows=len(samples), host_inclusive_ms=stages))
        result['diagnostics'].append(dict(fixture=observation['fixture'], active=observation['active'],
            ranks=ranks, scope='Host durations, parents contain children; do not sum or equate cpu() to copy cost'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(json.loads(args.input.read_text())), indent=2)+'\n')
