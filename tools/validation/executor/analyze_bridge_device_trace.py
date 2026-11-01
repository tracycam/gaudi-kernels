"""Attribute profiled step intervals using engine interval unions (not sums)."""
import argparse
import json
from pathlib import Path
import statistics
import re
from collections import defaultdict, deque


def engine_family(thread):
    """Physical engines only; ENG ARC tracks describe scheduler activity."""
    if re.fullmatch(r'(?:\[D\d+\]\s*)?TPC(?:\s*\d+)?', thread):
        return 'TPC'
    if re.fullmatch(r'(?:\[D\d+\]\s*)?MME(?:\s*\d+)?', thread):
        return 'MME'
    if 'Internal DMA' in thread and not thread.startswith('ENG ARC'):
        return 'EDMA'
    if re.fullmatch(r'DMA\s+\d+', thread):
        return 'PDMA'
    if re.fullmatch(r'Port\s+\d+', thread):
        return 'NIC'
    return None


def union(intervals):
    result = []
    for begin, end in sorted(intervals):
        if not result or begin > result[-1][1]:
            result.append([begin, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def from_sdk_jsonl(path):
    """Pair engine events by context and name; engines need not be nested stacks."""
    with path.open() as source:
        events = [json.loads(line) for line in source]
    threads = {(e['pid'],e['tid']):e['args']['name'] for e in events
               if e['ph']=='M' and e['name']=='thread_name'}
    pending = defaultdict(deque)
    normalized, unmatched = [], []
    for event in sorted(events, key=lambda e:e['ts']):
        phase = event['ph']
        if phase in ('B','E'):
            key = (event['pid'],event['tid'],event['id'],event['name'])
            if phase == 'B':
                pending[key].append(event)
                continue
            if not pending[key]:
                unmatched.append(event)
                continue
            start = pending[key].popleft()
            event = dict(start, ph='X', dur=event['ts']-start['ts'])
        elif phase not in ('X','M'):
            continue
        thread = threads.get((event['pid'],event['tid']),'')
        if event['ph']=='X' and engine_family(thread):
            event = dict(event, cat='kernel', args=dict(event['args'],
                         recipeId=event['args'].get('recipe_id')))
        normalized.append(event)
    unmatched.extend(event for queue in pending.values() for event in queue)
    return dict(traceEvents=normalized, unmatched_sdk_events=unmatched)


def analyze(document, *, skip_first=1):
    events = document['traceEvents']
    threads = {(e['pid'], e['tid']): e['args']['name'] for e in events
               if e.get('ph') == 'M' and e.get('name') == 'thread_name'}
    kernels = []
    for event in events:
        if event.get('ph') != 'X' or event.get('cat') not in ('kernel', 'hpu_op'):
            continue
        name = threads.get((event['pid'], event['tid']), '')
        family = engine_family(name)
        if family:
            kernels.append((event['ts'], event['ts']+event['dur'], family,
                            event.get('args', {}).get('recipeId')))
    parents = sorted([e for e in events if e.get('ph') == 'X' and
                      e['name'] == 'runner.sample_tokens'], key=lambda e:e['ts'])
    # The first call can begin before profiling starts. Keep complete subsequent
    # scopes only; these timings include profiler cost and are NOT model TPS.
    rows = []
    for parent in parents[skip_first:]:
        begin, end = parent['ts'], parent['ts']+parent['dur']
        inside = [(max(a, begin), min(b, end), family, recipe)
                  for a, b, family, recipe in kernels if a < end and b > begin]
        compute = [k for k in inside if k[2] in ('TPC', 'MME')]
        if not compute:
            continue
        lo, hi = min(k[0] for k in compute), max(k[1] for k in compute)
        groups = {family: union((a,b) for a,b,f,_ in inside if f == family)
                  for family in ('TPC','MME','EDMA','PDMA','NIC')}
        active = union((max(a,lo),min(b,hi)) for a,b,_,_ in inside if a<hi and b>lo)
        busy = lambda xs: sum(b-a for a,b in xs)
        # A compiled recipe ID is reused across layers; grouping all occurrences
        # by ID would incorrectly fill inter-layer gaps. Keep consecutive runs.
        runs = []
        for a,b,_,recipe in sorted(compute):
            if runs and runs[-1][2] == recipe:
                runs[-1][1] = max(runs[-1][1],b)
            else:
                runs.append([a,b,recipe])
        recipe_envelope = union((a,b) for a,b,_ in runs)
        rows.append(dict(begin_us=begin, end_us=end, parent_us=end-begin,
            compute_span_us=hi-lo, before_compute_us=lo-begin, after_compute_us=end-hi,
            compute_union_us=busy(union((a,b) for a,b,_,_ in compute)),
            observed_engine_idle_within_compute_span_us=hi-lo-busy(active),
            between_compute_recipe_runs_us=hi-lo-busy(recipe_envelope),
            recipe_runs=len(runs), engine_union_us={f:busy(xs) for f,xs in groups.items()}))
        for label, needle in (('sdk_launch','enqueueWithExternalEventsExt'), ('hccl','hcclAllGather')):
            calls = [e for e in events if e.get('ph')=='X' and needle in e['name'] and
                     begin <= e['ts'] and e['ts']+e.get('dur',0) <= end]
            rows[-1][label+'_calls'] = len(calls)
            rows[-1][label+'_us'] = sum(e.get('dur',0) for e in calls)
    if not rows:
        raise ValueError('No complete sample scopes with hardware compute events')
    medians = {key: statistics.median(row[key] for row in rows) for key in rows[0]
               if key not in ('begin_us','end_us','engine_union_us')}
    medians['engine_union_us'] = {kind: statistics.median(row['engine_union_us'][kind]
        for row in rows) for kind in rows[0]['engine_union_us']}
    return dict(steps=len(rows), median=medians, rows=rows,
        scope='Profiled hardware interval unions; not unprofiled TPS. Engine unions overlap. '
              'Missing NIC spans and timestamp-only DMA markers cannot establish all-engine idle. '
              'Recipe runs use consecutive observed IDs, not a launch count. '
              'Gaps include communication and dependencies, not proven CPU starvation.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sdk-jsonl',action='store_true')
    a=p.parse_args()
    document=from_sdk_jsonl(a.trace) if a.sdk_jsonl else json.loads(a.trace.read_text())
    r=analyze(document, skip_first=0 if a.sdk_jsonl else 1)
    if a.sdk_jsonl:
        r['unmatched_sdk_events']=len(document['unmatched_sdk_events'])
    a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['median'],indent=2))


if __name__ == '__main__':
    main()
