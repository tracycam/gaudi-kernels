"""Frontend token ITL, EngineCore, worker, and diagnostic device span."""
import json
from tools.consolidation.analyze_latency_repeat import analyze_arm
from tools.consolidation.compare_migration import quantile, steady


def returned_timing(row):
    window = steady(row)
    index = row['steps'].index(window[0])
    previous = next(s for s in reversed(row['steps'][:index]) if s['new_tokens'] > 0)
    outputs = [s for s in window if s['new_tokens'] > 0]
    if sum(s['new_tokens'] for s in outputs) != 128 or any(s['new_tokens'] != 1 for s in outputs):
        raise ValueError('Requires 128 actual B1 token return intervals')
    times = [previous['end_ns']]+[s['end_ns'] for s in outputs]
    intervals = [(b-a)/1e6 for a, b in zip(times, times[1:])]
    return {'tps': 128*1e9/(times[-1]-times[0]), 'itl_p50_ms': quantile(intervals, .5),
            'itl_p99_ms': quantile(intervals, .99), 'tokens': 128,
            'scope': 'Actual frontend token returns; stalled steps remain inside intervals'}


def analyze(result):
    arms = []
    for row in result['runs']:
        if row['execution_gate_pass'] is not True:
            raise ValueError('Execution/bitwise gate rejected')
        base = analyze_arm(row, context=result['context'], clock=result['clock_domain'])
        ranks = {r['rank']: {n['position']: n for n in r['native_steps']} for r in row['ranks']}
        core = {}
        emitted = 0
        for step in row['core_steps']:
            for output in step['tokens']:
                for token in output['tokens']:
                    emitted += 1
                    if token != row['ids'][emitted-1]:
                        raise ValueError('EngineCore/frontend token sequence differs')
                    core[emitted] = step
        if emitted != len(row['ids']):
            raise ValueError('EngineCore producer coverage incomplete')
        metrics = []
        for step in steady(row):
            if step['new_tokens'] != 1:
                continue
            producer = core[step['emitted']]
            pos = result['context']+step['emitted']-2
            ns = [ranks[r][pos] for r in range(8)]
            metrics.append({'core_wall_ms': (producer['end_ns']-producer['begin_ns'])/1e6,
                'max_worker_wall_ms': max(n['worker_wall_ns'] for n in ns)/1e6,
                'core_to_frontend_return_ms': (step['end_ns']-producer['end_ns'])/1e6,
                'max_native_wall_ms': max(n['native_wall_ns'] for n in ns)/1e6,
                'max_enqueue_ms': max(n['enqueue_ns'] for n in ns)/1e6,
                'max_native_after_enqueue_ms': max(n['native_wall_ns']-n['enqueue_ns'] for n in ns)/1e6,
                'max_worker_outside_native_ms': max(n['worker_wall_ns']-n['native_wall_ns'] for n in ns)/1e6,
                **({'max_device_stream_span_ms': max(n['device_stream_span_ns'] for n in ns)/1e6}
                    if all('device_stream_span_ns' in n for n in ns) else {})})
        statistics = {k: {'p50': quantile([m[k] for m in metrics], .5),
                          'p99': quantile([m[k] for m in metrics], .99)} for k in metrics[0]}
        arm = {k: row[k] for k in ('kind', 'cycle', 'arm', 'replay_mode')}
        arm.update(frontend=returned_timing(row), layers=statistics,
                   all_rank_tokens_matched=base['matched_native_steps'],
                   host_binding_verified=all(h['all_threads_local'] and h['module_match'] and h['memory_policy_mode']==2 for h in row['host_placement']),
                   cpu_threads=[len(h['threads']) for h in row['host_placement']],
                   api_detail_enabled=row['replay_mode']!='compact')
        arms.append(arm)
    pairs = []
    for cycle in sorted({r['cycle'] for r in arms if r['kind']=='runtime'}):
        group = [r for r in arms if r['kind']=='runtime' and r['cycle']==cycle]
        old, new = [[r for r in group if r['replay_mode']==mode] for mode in ('legacy','compact')]
        def mean(rows, key):
            return sum(r['frontend'][key] for r in rows)/len(rows)
        pairs.append({'cycle': cycle, 'legacy_mean_tps': mean(old,'tps'),
                      'compact_mean_tps': mean(new,'tps'),
                      'tps_delta_percent': 100*(mean(new,'tps')/mean(old,'tps')-1)})
    return {'arms': arms, 'pairs': pairs,
            'scope': 'Three clock boundaries plus worker/native subintervals. Device events measure stream span including queue starvation and collective waits; engine busy requires trace. Diagnostic modes excluded from paired throughput.'}
