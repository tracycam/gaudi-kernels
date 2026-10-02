#!/usr/bin/env python3
"""Compare actual physical compute node envelopes in sealed trace windows."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re


def normalized(op):
    return re.sub(r'^(fused_kernel_0x[0-9A-Fa-f]+)_[0-9A-Fa-f]+_','\\1_',op)


def parse(trace, summary):
    data=json.loads(summary.read_text());data=data[0] if isinstance(data,list) else data
    token=data['tokens'][-1];low,high=token['start_us'],token['end_us']
    pids=set();stack={};intervals=[]
    for line in trace.open():
        event=json.loads(line);args=event.get('args',{});pid=event['pid']
        if event['ph']=='M' and event['name']=='process_name' and args.get('name','').startswith(('*TPC (','*MME (')):pids.add(pid)
        if pid not in pids or not args.get('op') or args['op']=='null' or event['ph'] not in ('B','E'):continue
        if event['ts']<low-.001 or event['ts']>high+.001:continue
        key=(pid,event['tid'],event['name'],event['id'])
        if event['ph']=='B':assert key not in stack;stack[key]=event
        else:
            begin=stack.pop(key)
            intervals.append(dict(start=begin['ts'],end=event['ts'],name=event['name'],op=args['op'],recipe=args['recipe_id'],pid=pid,tid=event['tid']))
    assert not stack
    recipes=[];counts=collections.Counter()
    for span in token['recipe_compute_spans']:
        nodes={}
        for interval in intervals:
            if interval['recipe'] not in span['recipe_ids'] or interval['start']<span['start_us']-.001 or interval['end']>span['end_us']+.001:continue
            key=interval['name']
            if key not in nodes:nodes[key]=dict(name=key,op=interval['op'],normalized_op=normalized(interval['op']),start=interval['start'],end=interval['end'],core_intervals=0)
            node=nodes[key];node['start']=min(node['start'],interval['start']);node['end']=max(node['end'],interval['end']);node['core_intervals']+=1
        ordered=sorted(nodes.values(),key=lambda x:(x['start'],x['end'],x['name']))
        for node in ordered:node['duration_us']=node['end']-node['start'];counts[node['normalized_op']]+=1
        recipes.append(dict(**span,nodes=ordered,normalized_counts=dict(collections.Counter(n['normalized_op'] for n in ordered))))
    assert sum(counts.values())==token['distinct_compute_node_names_per_launch'],(sum(counts.values()),token['distinct_compute_node_names_per_launch'])
    return dict(trace=str(trace),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
        token={k:v for k,v in token.items() if k!='recipe_compute_spans'},recipes=recipes,normalized_counts=dict(counts))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old_trace','old_summary','new_trace','new_summary'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    old=parse(a.old_trace,a.old_summary);new=parse(a.new_trace,a.new_summary)
    assert len(old['recipes'])==len(new['recipes'])
    deltas={k:new['normalized_counts'].get(k,0)-old['normalized_counts'].get(k,0) for k in set(old['normalized_counts'])|set(new['normalized_counts'])}
    pairs=[]
    for left,right in zip(old['recipes'],new['recipes']):
        assert left['launch_in_token']==right['launch_in_token']
        delta={k:right['normalized_counts'].get(k,0)-left['normalized_counts'].get(k,0) for k in set(left['normalized_counts'])|set(right['normalized_counts'])}
        pairs.append(dict(launch_in_token=left['launch_in_token'],old_recipe_ids=left['recipe_ids'],new_recipe_ids=right['recipe_ids'],
            old_count=len(left['nodes']),new_count=len(right['nodes']),node_delta=len(right['nodes'])-len(left['nodes']),
            old_internal_no_compute_us=left['internal_no_compute_us'],new_internal_no_compute_us=right['internal_no_compute_us'],
            op_deltas={k:v for k,v in sorted(delta.items()) if v}))
    categories=collections.defaultdict(lambda:dict(recipes=0,old_internal_no_compute_us=0.,new_internal_no_compute_us=0.))
    for left,right,pair in zip(old['recipes'],new['recipes'],pairs):
        delta=pair['op_deltas']
        if delta.get('cast_f32_to_bf16',0)>0:category='new_explicit_cast_before_inputnorm'
        elif any(n['op'].startswith('gk_moe_gp_') for n in right['nodes']):category='postattention_norm_and_moe'
        elif pair['node_delta']==-3:category='first_two_norm_bundles'
        elif delta.get('abs_fwd_bf16',0)>0:category='first_dense_postattention'
        else:category='other'
        item=categories[category];item['recipes']+=1
        item['old_internal_no_compute_us']+=left['internal_no_compute_us'];item['new_internal_no_compute_us']+=right['internal_no_compute_us']
    for item in categories.values():item['delta_internal_no_compute_us']=item['new_internal_no_compute_us']-item['old_internal_no_compute_us']
    report=dict(recipe_categories=dict(categories),old=old,new=new,op_deltas={k:v for k,v in sorted(deltas.items()) if v},recipe_pairs=pairs,
        scope='read-only cross-run last-token physical compute nodes, normalized only compile-order suffix in fused GUID; temporal differences are not causal ablations')
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(old_nodes=sum(old['normalized_counts'].values()),new_nodes=sum(new['normalized_counts'].values()),op_deltas=report['op_deltas'],
        recipe_delta_histogram=dict(collections.Counter(p['node_delta'] for p in pairs))),indent=2))

if __name__=='__main__':main()
