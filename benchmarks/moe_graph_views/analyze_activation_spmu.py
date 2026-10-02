"""Summarize decoded C values in conservative whole interior sample windows.

This deliberately does not infer a TPC clock, add overlapping stall categories,
or call a counter sample a hardware cycle count. Chrome C values are step
signals: unchanged values carry forward between emitted changes.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
events=json.loads(a.trace.read_text())['traceEvents']
counter_events=defaultdict(list);labels={};core_names={}
for e in events:
    if e['ph']=='M' and e['name']=='thread_name' and 'TPC ' in e['args'].get('name',''):
        core_names[e['tid']]=e['args']['name']
    if e['ph']=='C' and e.get('cat')=='SPMU':
        index=int(re.match(r'\[(\d+)\]',e['name'])[1]);labels[index]=e['name']
        value=e['args'][e['name']];counter_events[e['tid']].append((e['ts'],index,value))
groups={};gaps=[]
for tid,records in counter_events.items():
    records.sort();buckets=[]
    for t,index,value in records:
        if not buckets or t-buckets[-1]['start']>.2:buckets.append({'start':t,'end':t,'updates':{}})
        buckets[-1]['end']=t;buckets[-1]['updates'][index]=value
    state={i:0 for i in labels}
    for bucket in buckets:
        state.update(bucket['updates']);bucket['values']=state.copy()
    for previous,current in zip(buckets,buckets[1:]):
        gap=current['start']-previous['start']
        if 1<gap<10:gaps.append(gap)
    groups[tid]=buckets
period=statistics.median(gaps)
stack={};nodes=defaultdict(list)
for e in events:
    args=e.get('args',{});hw=args.get('HW event name');key=(e['pid'],e['tid'])
    if e['ph']=='B' and hw=='TPC_SPU_START':stack[key]=e
    elif e['ph']=='E' and hw=='TPC_SPU_START_TO_SPU_HALT':
        start=stack.pop(key);op=args['op']
        if op in ['ub_direct_down','downact_direct_down_broadcast','diag_direct_gp_broadcast']:
            nodes[(op,e['name'])].append((e['tid'],start['ts'],e['ts']))
assert not stack
result={'trace_sha256':hashlib.sha256(a.trace.read_bytes()).hexdigest(),'raw_counter_C_events':sum(len(v) for v in counter_events.values()),
        'requested_capture_window_cycles':1024,'observed_median_update_gap_us':period,
        'decoded_counter_names':labels,'counter_21':'No C events; .spmu.res reports zero. Not interpreted as absence of stalls.',
        'scope':'Forward-filled decoded C sample values, whole interior windows only. Raw units; not divided by requested1024, not added as time.',
        'selection':'Within one core SPU interval, adjacent emitted sample groups spaced median±0.2us; preceding group starts>=node start+0.1us, current group ends<=node end-0.1us.',
        'nodes':[]}
for (op,name),intervals in nodes.items():
    samples=[];covered=0;per_interval=[]
    for tid,start,end in intervals:
        count=0
        for previous,current in zip(groups.get(tid,[]),groups.get(tid,[])[1:]):
            gap=current['start']-previous['start']
            if previous['start']>=start+.1 and current['end']<=end-.1 and abs(gap-period)<=.2:
                samples.append(current['values']);covered+=gap;count+=1
        per_interval.append(count)
    record={'op':op,'name':name,'core_intervals':len(intervals),'active_cores':len({v[0] for v in intervals}),
            'interior_sample_windows':len(samples),'interior_window_coverage_fraction':covered/sum(end-start for _,start,end in intervals),
            'samples_per_core_interval_range':[min(per_interval),max(per_interval)],'raw_sample_values':{}}
    for index in labels:
        values=[v[index] for v in samples]
        record['raw_sample_values'][index]={'mean':statistics.mean(values),'median':statistics.median(values),'min':min(values),'max':max(values)}
    result['nodes'].append(record)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
