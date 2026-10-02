"""Pair native engine B/E events by recipe invocation, without CSV attribution.

The analyzer CSV may coalesce shared kernels into another graph's rows. Native
recipe id/idx and physical node ids retain each invocation's complete identity.
No host intervals, profiler recipe-duration fields, or ideal-time subtraction
are used. The emitted envelope includes physical inter-node gaps.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def analyze(trace, postgraphs, names, operation_aliases=None):
    events=json.loads(trace.read_text())['traceEvents']
    declared={}
    files=sorted(postgraphs.rglob('*.json')) if postgraphs.is_dir() else [postgraphs]
    for path in files:
        for graph in json.loads(path.read_text())['graphs']:
            name=graph['name']
            if name in names:
                # Physical compilation output; logical views have no engine event.
                aliases={'memcpy_dma':'DmaMemcpy', **(operation_aliases or {})}
                declared[name]={int(node['id']):aliases.get(node['guid'],node['guid']) for node in graph['nodes']
                                if node.get('engine') in ('TPC','DMA','MME')}
    paired=defaultdict(lambda:defaultdict(list))
    for event in events:
        args=event.get('args',{})
        if args.get('Recipe name') not in names or event.get('ph') not in ('B','E'):
            continue
        key=(args['Recipe name'],args['recipe id'],args['recipe idx'],
             args['Unique Node ID'],event['pid'],event['tid'],event['id'])
        paired[key][event['ph']].append(event)
    invocations=defaultdict(list)
    for key,phases in paired.items():
        starts=sorted(phases['B'],key=lambda e:e['ts'])
        ends=sorted(phases['E'],key=lambda e:e['ts'])
        if len(starts)!=len(ends):
            raise ValueError(f'unpaired native events: {key}')
        for index,(start,end) in enumerate(zip(starts,ends)):
            if start['ts']>end['ts'] or (index and start['ts']<ends[index-1]['ts']):
                raise ValueError(f'invalid engine ordering: {key}')
            invocations[key[:3]].append((start,end))
    rows=[]
    for (name,recipe_id,recipe_idx),pairs in sorted(invocations.items(),key=lambda x:x[0][2]):
        nodes=defaultdict(list)
        for start,end in pairs:nodes[start['args']['Unique Node ID']].append((start,end))
        if name not in declared or set(nodes)!=set(declared[name]):
            raise ValueError(f'incomplete physical node coverage: {name} invocation {recipe_idx}')
        for node_id,intervals in nodes.items():
            if any(b['args']['op']!=declared[name][node_id] for b,e in intervals):
                raise ValueError(f'native/PostGraph operation mismatch: {name} node {node_id}')
        operations=defaultdict(int)
        for intervals in nodes.values():operations[intervals[0][0]['args']['op']]+=1
        start_us=min(b['ts'] for b,e in pairs);end_us=max(e['ts'] for b,e in pairs)
        detail=[]
        for node_id,intervals in nodes.items():
            begins=[b['ts'] for b,e in intervals];ends=[e['ts'] for b,e in intervals]
            detail.append({'node_id':node_id,'op':intervals[0][0]['args']['op'],
                           'start_offset_us':min(begins)-start_us,'end_offset_us':max(ends)-start_us,
                           'envelope_us':max(ends)-min(begins),
                           'engine_count':len({(b['pid'],b['tid']) for b,e in intervals}),
                           'engine_intervals':[[b['pid'],b['tid'],b['ts']-start_us,e['ts']-start_us]
                                               for b,e in intervals]})
        rows.append({'recipe':name,'recipe_id':recipe_id,'recipe_idx':recipe_idx,
                     'start_us':start_us,'end_us':end_us,'envelope_us':end_us-start_us,
                     'physical_nodes':len(nodes),'operations':dict(operations),
                     'nodes':sorted(detail,key=lambda x:x['start_offset_us'])})
    if not rows:raise ValueError('no selected physical events')
    return {'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
            'scope':'Physical engine first-start to last-completion per native recipe id/idx; includes inter-node gaps. No host/ideal subtraction. Incomplete profile capture limits sample count.',
            'invocations':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--postgraphs',type=Path,required=True)
    parser.add_argument('--recipe',action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.trace,args.postgraphs,args.recipe)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for name in args.recipe:
        rows=[r for r in result['invocations'] if r['recipe']==name]
        print(name,[(r['recipe_idx'],round(r['envelope_us'],3),r['physical_nodes']) for r in rows])
