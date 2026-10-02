#!/usr/bin/env python3
"""Read one sealed native trace: QKV and reducer envelopes, no cycle projection."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
compute={};stack={};intervals=[]
for line in a.trace.open():
 e=json.loads(line);args=e.get('args',{});pid=e['pid']
 if e['ph']=='M' and e['name']=='process_name' and args.get('name','').startswith(('*TPC (','*MME (')):
  compute[pid]='MME' if args['name'].startswith('*MME (') else 'TPC'
 match=re.search(r'model/(\d+)/self_attn/qkv_proj/',e['name'])
 if pid not in compute or not match or e['ph'] not in ('B','E'):continue
 key=(pid,e['tid'],e['name'],e['id'])
 if e['ph']=='B':assert key not in stack;stack[key]=e
 elif key in stack:
  start=stack.pop(key)
  intervals.append(dict(start=start['ts'],end=e['ts'],name=e['name'],op=args['op'],source_layer_label=int(match[1]),recipe=args['recipe_id'],engine=compute[pid],pid=pid,tid=e['tid']))
assert not stack
nodes=[];groups=[]
# Compiled names retain the first source-layer label across recipe reuse.
# Order the actual invocations globally, not by that cached label.
for x in sorted((v for v in intervals if v['op']=='gk_neumaier_isa_handschedule_v1'),key=lambda v:v['start']):
 if not groups or x['start']>groups[-1]['end']:
  groups.append(dict(start=x['start'],end=x['end'],cores={(x['pid'],x['tid'])},recipe=x['recipe'],source_layer_label=x['source_layer_label']))
 else:
  assert groups[-1]['recipe']==x['recipe']
  groups[-1]['end']=max(groups[-1]['end'],x['end']);groups[-1]['cores'].add((x['pid'],x['tid']))
assert len(groups)==210,len(groups)
previous=-float('inf')
for index,g in enumerate(groups):
 used=[x for x in intervals if x['recipe']==g['recipe'] and x['start']>previous and x['end']<=g['end']+.001]
 mme=[x for x in used if x['engine']=='MME'];assert mme
 nodes.append(dict(layer_ordinal=index%70,chronological_token=index//70,source_layer_label=g['source_layer_label'],
     reducer_start_us=g['start'],reducer_end_us=g['end'],reducer_us=g['end']-g['start'],active_cores=len(g['cores']),
     qkv_first_start_us=min(x['start'] for x in used),qkv_end_us=g['end'],qkv_envelope_us=g['end']-min(x['start'] for x in used),
     mme_envelope_us=max(x['end'] for x in mme)-min(x['start'] for x in mme),
     mme_to_reducer_gap_us=g['start']-max(x['end'] for x in mme),ops=sorted(set(x['op'] for x in used))))
 previous=g['end']
chunks=[]
for token in range(3):
 rows=sorted([x for x in nodes if x['chronological_token']==token],key=lambda x:x['reducer_start_us'])
 assert len(rows)==70 and all(x['reducer_end_us']<=y['qkv_first_start_us'] for x,y in zip(rows,rows[1:]))
 chunks.append(dict(token=token,nodes=70,reducer_sum_us=sum(x['reducer_us'] for x in rows),
  reducer_median_us=statistics.median(x['reducer_us'] for x in rows),reducer_min_us=min(x['reducer_us'] for x in rows),
  reducer_max_us=max(x['reducer_us'] for x in rows),qkv_envelope_sum_us=sum(x['qkv_envelope_us'] for x in rows),
  mme_median_us=statistics.median(x['mme_envelope_us'] for x in rows),
  mme_to_reducer_gap_median_us=statistics.median(x['mme_to_reducer_gap_us'] for x in rows)))
report=dict(trace=str(a.trace),trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),chunks=chunks,nodes=nodes,
 scope='actual rank0 production70-c three-token trace; cross-core node envelopes, not sum-of-core time; reducer time is a disjoint opportunity budget, not a predicted saving')
assert not a.out.exists();a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(chunks,indent=2))
