"""Align actual tokens and same-host rank intervals; do not invent GPU time."""
import argparse
import json
from pathlib import Path

from tools.consolidation.compare_migration import quantile, steady, timing


def analyze_arm(row, *, context, clock):
    ranks = row['ranks']
    if len(ranks) != 8 or {r['rank'] for r in ranks} != set(range(8)):
        raise ValueError('Eight distinct TP ranks are required')
    if any(r['clock_domain'] != clock for r in ranks):
        raise ValueError('Rank clocks do not share the coordinator host/boot/implementation')
    indexed = {}
    for rank in ranks:
        entries = {}
        for native in rank['native_steps']:
            position = native['position']
            if position in entries:
                raise ValueError('Duplicate native position')
            if native['caller_end_ns'] - native['caller_begin_ns'] != native['caller_ns']:
                raise ValueError('Inconsistent caller interval')
            if not 0 <= native['enqueue_ns'] <= native['native_wall_ns'] <= native['caller_ns']:
                raise ValueError('Invalid nested host durations')
            if native['synlaunch_ns'] + native['hccl_ns'] > native['enqueue_ns']:
                raise ValueError('API subintervals exceed enqueue duration')
            entries[position] = native
        indexed[rank['rank']] = entries
    aligned = []
    previous_output_end = {}
    prior_end = None
    for step in row['steps']:
        if step['new_tokens'] == 1:
            previous_output_end[step['emitted']] = prior_end
            prior_end = step['end_ns']
    for step in steady(row):
        entry = {'emitted': step['emitted'], 'new_tokens': step['new_tokens'],
                 'coordinator_ms': step['ms'], 'native_match': False}
        # Stalls and bridge fallback remain in coordinator timing. Neither is
        # assigned the preceding native token's interval.
        if step['new_tokens'] != 1:
            aligned.append(entry)
            continue
        position = context + step['emitted'] - 2
        native = [indexed[rank].get(position) for rank in range(8)]
        if any(n is None for n in native):
            entry.update(position=position, reason='no all-rank native interval')
            aligned.append(entry)
            continue
        token = row['ids'][step['emitted'] - 1]
        if any(n['token'] != token for n in native):
            raise ValueError('Native/coordinator token mismatch')
        begin = min(n['caller_begin_ns'] for n in native)
        end = max(n['caller_end_ns'] for n in native)
        # EngineCore can run ahead of the frontend's blocking step(), and
        # rank0 can publish output before other ranks finish host bookkeeping.
        # Equal clock domains do NOT imply nested intervals.
        entry.update(position=position, token=token, native_match=True,
            first_caller_begin_offset_ms=(begin-step['begin_ns'])/1e6,
            caller_envelope_ms=(end-begin)/1e6,
            last_caller_end_offset_ms=(end-step['end_ns'])/1e6,
            rank0_return_to_frontend_ms=(step['end_ns']-indexed[0][position]['caller_end_ns'])/1e6,
            frontend_caller_overlap_ms=max(0, min(end, step['end_ns'])-max(begin, step['begin_ns']))/1e6,
            rank_start_skew_ms=(max(n['caller_begin_ns'] for n in native)-begin)/1e6,
            rank_end_skew_ms=(end-min(n['caller_end_ns'] for n in native))/1e6,
            max_caller_ms=max(n['caller_ns'] for n in native)/1e6,
            max_native_wall_ms=max(n['native_wall_ns'] for n in native)/1e6,
            max_enqueue_ms=max(n['enqueue_ns'] for n in native)/1e6,
            max_native_wall_minus_enqueue_ms=max(n['native_wall_ns']-n['enqueue_ns'] for n in native)/1e6)
        if all(n.get('api_detail_valid', True) for n in native):
            entry.update(max_synlaunch_ms=max(n['synlaunch_ns'] for n in native)/1e6,
                         max_hccl_api_ms=max(n['hccl_ns'] for n in native)/1e6)
        previous_end = previous_output_end[step['emitted']]
        previous_native = indexed[0].get(position-1)
        if previous_end is not None and previous_native is not None:
            # Exact accounting of emitted-token intervals, including changes
            # in producer->consumer delay. This is not an engine-time split.
            old_delay = previous_end-previous_native['caller_end_ns']
            new_delay = step['end_ns']-indexed[0][position]['caller_end_ns']
            entry.update(frontend_return_itl_ms=(step['end_ns']-previous_end)/1e6,
                rank0_return_itl_ms=(indexed[0][position]['caller_end_ns']-previous_native['caller_end_ns'])/1e6,
                rank0_to_frontend_delay_change_ms=(new_delay-old_delay)/1e6)
        aligned.append(entry)
    matched = [s for s in aligned if s['native_match']]
    fields = sorted(set.intersection(*(set(s) for s in matched))) if matched else []
    fields = [k for k in fields if k.endswith('_ms')]
    return {'kind': row['kind'], 'cycle': row['cycle'], 'arm': row['arm'], 'installer': row['installer'], 'timing': timing([row]),
            'matched_native_steps': len(matched), 'unmatched_steps': len(aligned)-len(matched),
            'components': {key: {'p50': quantile([s[key] for s in matched], .5),
                                 'p99': quantile([s[key] for s in matched], .99)} for key in fields},
            'worst_steps': sorted(aligned, key=lambda s: s['coordinator_ms'], reverse=True)[:10],
            'aligned_steps': aligned}


def analyze(result):
    rows = result['runs']
    if not rows or not all(r.get('execution_gate_pass') is True for r in rows):
        raise ValueError('Every execution/bitwise/coverage gate must pass')
    arms = [analyze_arm(r, context=result['context'], clock=result['clock_domain']) for r in rows]
    cycles = []
    for cycle in sorted({r['cycle'] for r in rows if r['kind'] == 'paired'}):
        pair = [r for r in rows if r['kind'] == 'paired' and r['cycle'] == cycle]
        names = [r['installer'] for r in pair]
        if names not in (['historical', 'canonical', 'canonical', 'historical'],
                         ['canonical', 'historical', 'historical', 'canonical']):
            raise ValueError('Missing ABBA/BAAB cycle')
        old = timing([r for r in pair if r['installer'] == 'historical'])
        new = timing([r for r in pair if r['installer'] == 'canonical'])
        cycles.append({'cycle': cycle, 'order': names, 'historical': old, 'canonical': new,
                       'pass': new['tps'] >= old['tps']*.99 and new['p99_ms'] <= old['p99_ms']})
    sham = [r for r in rows if r['kind'] == 'sham']
    negative = None
    if sham:
        if len(sham) != 4 or any(r['installer'] != 'historical' for r in sham):
            raise ValueError('Sham control must use four historical arms')
        outer, inner = timing([sham[0], sham[3]]), timing([sham[1], sham[2]])
        negative = {'outer': outer, 'inner': inner,
                    'p99_inner_minus_outer_ms': inner['p99_ms']-outer['p99_ms'],
                    'scope': 'Identical installer; detects temporal variation, not a relaxed gate'}
    return {'cycles': cycles, 'all_paired_cycles_pass': bool(cycles) and all(c['pass'] for c in cycles),
            'sham': negative, 'arms': arms,
            'scope': 'Same-host monotonic signed offsets and exact returned-token interval accounting. '
                     'EngineCore executes independently of frontend step(); rank0 may publish before other ranks return. '
                     'Caller intervals are not assumed to be contained by frontend steps. Enqueue overlaps device execution; '
                     'Native wall minus enqueue includes synchronization wait and small staging/commit work, not total compute time. '
                     'Component percentiles are not additive. All stalls/fallbacks remain in end-to-end timing. '
                     'Targeted diagnostic only; does not replace full-model numerical admission.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(json.loads(args.result.read_text()))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'arms'}, indent=2))
