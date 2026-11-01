"""Pair CPU bridge spans and locate queued submission; durations are inclusive."""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(values):
    return dict(count=len(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values)) if values else dict(count=0)


def spans(document):
    stacks = defaultdict(list)
    result, unmatched = [], []
    # Bridge events come from multiple threads. Pair each thread independently;
    # complete Kineto X events do not participate in the B/E stack.
    for event in sorted(document['traceEvents'], key=lambda row: row.get('ts', 0)):
        key = (event.get('pid'), event.get('tid'))
        phase = event.get('ph')
        if phase == 'B':
            stacks[key].append(event)
        elif phase == 'E':
            if not stacks[key]:
                unmatched.append(dict(reason='end_without_begin', event=event))
                continue
            start = stacks[key].pop()
            if event.get('name') and event['name'] != start['name']:
                unmatched.append(dict(reason='mismatched_name', begin=start, end=event))
                continue
            result.append(dict(name=start['name'], pid=key[0], tid=key[1],
                               begin_us=start['ts'], end_us=event['ts'], args=start.get('args', {})))
        elif phase == 'X':
            result.append(dict(name=event['name'], pid=key[0], tid=key[1],
                               begin_us=event['ts'], end_us=event['ts']+event.get('dur', 0),
                               args=event.get('args', {})))
    for pending in stacks.values():
        unmatched.extend(dict(reason='begin_without_end', event=event) for event in pending)
    return result, unmatched


def analyze(document, *, command_details=False):
    intervals, unmatched = spans(document)
    groups = defaultdict(list)
    for row in intervals:
        groups[row['name']].append(row)
    launch_stages = ('LaunchMemoryPreflight', 'LaunchAddressLease', 'LaunchSynapseSubmit')
    indexed_rows = defaultdict(list)
    for name in (*launch_stages, 'PreparedRegionSubmit', *(('launch',) if command_details else ())):
        for row in groups[name]:
            indexed_rows[(name, row['pid'], row['tid'])].append(row)
    interval_index = {}
    for key, rows in indexed_rows.items():
        rows.sort(key=lambda row: row['begin_us'])
        interval_index[key] = ([row['begin_us'] for row in rows], rows)

    def contained(name, parent):
        starts, rows = interval_index.get((name, parent['pid'], parent['tid']), ([], []))
        first = bisect_left(starts, parent['begin_us'])
        last = bisect_right(starts, parent['end_us'])
        return [r for r in rows[first:last] if r['end_us'] <= parent['end_us']]
    selected = ('HPUGraph.replay', 'NativePlanReplay', 'NativePlanBindingPreflight',
                'SubmitCollectivePlan', 'NativePlanSubmit', 'PreparedRegionSubmit', 'Launch', 'launch',
                'LaunchMemoryPreflight', 'LaunchAddressLease', 'LaunchSynapseSubmit')
    stages = {name: dict(threads=sorted({r['tid'] for r in groups[name]}),
                       inclusive_us=summarize([r['end_us']-r['begin_us'] for r in groups[name]]))
              for name in selected}
    submissions = []
    for worker in groups['NativePlanSubmit']:
        callers = [r for r in groups['SubmitCollectivePlan'] if r['pid'] == worker['pid'] and
                   r['begin_us'] <= worker['begin_us'] and r['end_us'] >= worker['end_us']]
        if len(callers) != 1:
            # Compute-only plans submit on the caller and have no HCCL queue.
            continue
        caller = callers[0]
        launches = [r for r in groups['Launch'] if r['pid'] == worker['pid'] and
                    r['tid'] == worker['tid'] and r['begin_us'] >= worker['begin_us'] and
                    r['end_us'] <= worker['end_us']]
        submissions.append(dict(caller_tid=caller['tid'], submission_tid=worker['tid'],
            caller_begin_us=caller['begin_us'], caller_end_us=caller['end_us'],
            submit_begin_us=worker['begin_us'], submit_end_us=worker['end_us'],
            queue_entry_to_submit_us=worker['begin_us']-caller['begin_us'],
            submit_us=worker['end_us']-worker['begin_us'],
            submit_end_to_caller_return_us=caller['end_us']-worker['end_us'],
            launcher_count=len(launches),
            launcher_inclusive_us=sum(r['end_us']-r['begin_us'] for r in launches)))
        submissions[-1]['launch_stages'] = {
            name: dict(count=len(rows), inclusive_us=sum(r['end_us']-r['begin_us'] for r in rows))
            for name in launch_stages for rows in (contained(name, worker),)}
        prepared = contained('PreparedRegionSubmit', worker)
        submissions[-1]['prepared_regions'] = dict(count=len(prepared),
            inclusive_us=sum(r['end_us']-r['begin_us'] for r in prepared))
        if command_details:
            commands = []
            for index, launch in enumerate(sorted(launches, key=lambda r:r['begin_us'])):
                compute = contained('launch', launch)
                commands.append(dict(index=index, begin_us=launch['begin_us'],
                    end_us=launch['end_us'], launcher_us=launch['end_us']-launch['begin_us'],
                    recipe_debug_index=launch['args'].get('index'),
                    compute_launch_us=sum(r['end_us']-r['begin_us'] for r in compute),
                    compute_launch_count=len(compute)))
            submissions[-1]['commands'] = commands
    return dict(base_time_ns=document.get('baseTimeNanoseconds'),
        stages=stages, queued_submissions=submissions,
        queued_summary={name:summarize([r[name] for r in submissions]) for name in
                        ('queue_entry_to_submit_us','submit_us','submit_end_to_caller_return_us')},
        unmatched=unmatched,
        scope='CPU-only diagnostic wall durations with profiling overhead; nested stages overlap. '
              'SDK calls may include device backpressure. These are not device compute or model TPS.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('traces', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--command-details', action='store_true')
    args = parser.parse_args()
    result = {str(path):analyze(json.loads(path.read_text()), command_details=args.command_details)
              for path in args.traces}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for path, data in result.items():
        print(json.dumps(dict(trace=path, queued=data['queued_summary'],
                              unmatched=len(data['unmatched']))))


if __name__ == '__main__':
    main()
