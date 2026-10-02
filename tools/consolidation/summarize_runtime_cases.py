#!/usr/bin/env python3
"""Summarize NUMA placement and replay timing without dropping slow arms."""
import argparse
import json
from pathlib import Path

from tools.consolidation.analyze_runtime import analyze
from tools.consolidation.compare_migration import quantile


def summarize(path):
    result = json.loads(path.read_text())
    if result['status'] != 'DIAGNOSTIC' or len(result['runs']) != 10:
        raise ValueError('Requires eight throughput arms and two diagnostics')
    report = analyze(result)
    modes = {}
    for mode in ('legacy', 'compact', 'detail', 'device'):
        arms = [row for row in report['arms'] if row['replay_mode'] == mode]
        modes[mode] = {'arms': len(arms),
            'returned_token_tps': len(arms)/sum(1/a['frontend']['tps'] for a in arms),
            'arm_tps': [a['frontend']['tps'] for a in arms],
            'median_of_arm_p50_ms': {key: quantile([a['layers'][key]['p50'] for a in arms], .5)
                for key in arms[0]['layers']},
            'arm_frontend_itl_p99_ms': [a['frontend']['itl_p99_ms'] for a in arms]}
    placements = []
    submissions = set()
    captured_counts = set()
    for row in result['runs']:
        for rank in row['ranks']:
            for capture in rank['captures']:
                submissions.add(capture['submission_count'])
                captured_counts.add(capture['plan_info'][2])
        for snapshot in row['host_placement']:
            node = 'N'+str(snapshot['expected']['numa_node'])
            pinned = snapshot['pinned_staging']
            def local(buffer):
                mapping = buffer['mapping']
                return bool(mapping and mapping['pages']) and set(mapping['pages']) == {node}
            placements.append({'rank': snapshot['rank'], 'mode': row['replay_mode'],
                'physical_device_match': snapshot['device_match'] and snapshot['module_match'],
                'all_threads_local': snapshot['all_threads_local'],
                'memory_policy_mode': snapshot['memory_policy_mode'],
                'thread_count': len(snapshot['threads']), 'pinned_buffers': len(pinned),
                'pinned_buffers_with_only_local_pages': sum(local(buffer) for buffer in pinned)})
    ranks = []
    for rank in range(8):
        observations = [row for row in placements if row['rank'] == rank]
        def span(key):
            return [min(row[key] for row in observations), max(row[key] for row in observations)]
        ranks.append({'rank': rank, 'observed_arms': len(observations),
            'physical_device_verified_every_arm': all(row['physical_device_match'] for row in observations),
            'cpu_local_every_arm': all(row['all_threads_local'] for row in observations),
            'memory_policy_modes': sorted({row['memory_policy_mode'] for row in observations}),
            'thread_count_range': span('thread_count'), 'pinned_buffer_count_range': span('pinned_buffers'),
            'local_pinned_buffer_count_range': span('pinned_buffers_with_only_local_pages'),
            'pinned_local_every_arm': all(row['pinned_buffers'] > 0 and
                row['pinned_buffers_with_only_local_pages'] == row['pinned_buffers'] for row in observations)})
    supervisor = json.loads((path.parent.parent/(path.parent.name+'.exit.json')).read_text())
    return {'case': path.parent.name, 'measured_source': supervisor['source_commit'],
        'binding': result['selection']['engine']['runtime']['host']['numa_binding'],
        'all_ten_sequences_match': all(row['execution_gate_pass'] for row in result['runs']),
        'effective_submission_counts': sorted(submissions),
        'captured_record_counts': sorted(captured_counts),
        'throughput_modes': modes, 'paired_cycles': report['pairs'],
        'compact_vs_legacy_tps_percent': 100*(modes['compact']['returned_token_tps']/modes['legacy']['returned_token_tps']-1),
        'placement': ranks,
        'scope': 'Full70 TP8, B1 greedy, 4096 prompt tokens, 160 generated tokens/arm; mature128 actual-return intervals. No discarded arms. Diagnostic detail/device modes excluded from paired throughput. NUMA cases use separate processes; clock-layer medians are not an additive decomposition. Reuses prior full numerical qualification; new token/capture equivalence checked here.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unbound', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = [summarize(args.unbound), summarize(args.local)]
    if [case['binding'] for case in cases] != ['none', 'local']:
        raise ValueError('Expected unbound/local placement cases')
    deltas = {mode: 100*(cases[1]['throughput_modes'][mode]['returned_token_tps']/
                         cases[0]['throughput_modes'][mode]['returned_token_tps']-1)
              for mode in ('legacy', 'compact')}
    args.output.write_text(json.dumps({'cases': cases, 'observed_local_vs_unbound_tps_percent': deltas,
        'scope': 'One resident process per placement, two interleaved replay cycles per process. Placement delta is an observed cross-process comparison, not isolated statistical proof of NUMA speedup.'}, indent=2)+'\n')
