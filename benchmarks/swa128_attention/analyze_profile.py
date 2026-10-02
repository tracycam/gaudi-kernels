"""Summarize native engine envelopes without summing overlapping engine time.

The profiler CSV has one row per node invocation. We group equal-count repeated
nodes by their chronological ordinal, then assert invocation envelopes do not
overlap. This requires a serialized captured-recipe replay, as in device.py.
"""
import argparse
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path


def union_us(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:
            merged[-1][1]=max(merged[-1][1],end)
        else:
            merged.append([start,end])
    return sum(end-start for start,end in merged)


def analyze(path):
    graphs=collections.defaultdict(lambda:collections.defaultdict(list))
    for row in csv.DictReader(path.open()):
        if row['Unit'] not in ('TPC','MME','EDMA','DMA'):
            raise ValueError('unhandled physical engine unit: '+row['Unit'])
        graphs[row['Graph Name']][row['Unique node id']].append(row)
    result={}
    for name,nodes in graphs.items():
        counts={len(rows) for rows in nodes.values()}
        if len(counts)!=1:
            raise ValueError('unequal node invocation counts: '+name)
        count=counts.pop()
        if count<2:
            continue
        for rows in nodes.values():
            rows.sort(key=lambda r:float(r['Start time of node']))
        invocations=[]
        last_end=None
        for i in range(count):
            rows=[rs[i] for rs in nodes.values()]
            intervals=[(float(r['Start time of node']),float(r['End time of node'])) for r in rows]
            start=min(a for a,b in intervals);end=max(b for a,b in intervals)
            if last_end is not None and start<last_end:
                raise ValueError('overlapping invocations require trace recipe-index grouping')
            last_end=end
            busy=union_us(intervals)
            invocations.append({'ordinal':i,'start_us':start,'end_us':end,'envelope_us':end-start,
                                'node_interval_union_us':busy,'internal_uncovered_us':end-start-busy,
                                'unit_interval_union_us':{
                                    unit:union_us([(float(r['Start time of node']),float(r['End time of node'])) for r in rows if r['Unit']==unit])
                                    for unit in sorted({r['Unit'] for r in rows})}})
        def summary(rows):
            return {key:statistics.median(r[key] for r in rows)
                    for key in ('envelope_us','node_interval_union_us','internal_uncovered_us')}
        result[name]={'invocation_count':count,'nodes_per_invocation':len(nodes),
                      'units_per_invocation':dict(collections.Counter(rs[0]['Unit'] for rs in nodes.values())),
                      'ops_per_invocation':dict(collections.Counter(rs[0]['Op Type'] for rs in nodes.values())),
                      'all_median':summary(invocations),'exclude_first_median':summary(invocations[1:]),
                      'invocations':invocations}
    return {'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'graphs':result,
            'scope':'Native first-engine-start to last-engine-end envelope. Node intervals are unioned including EDMA; this is not useful-ALU time, launch count, physical HBM traffic, or model TPS. Profiled host/event timing is not used.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(analyze(a.csv),indent=2)+'\n')
