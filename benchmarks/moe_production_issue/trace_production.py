"""Actual native trace intervals; core union is not sum-of-core elapsed time."""
import argparse,collections,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
starts={};intervals=collections.defaultdict(list);names=collections.defaultdict(set)
for line in a.trace.open():
    e=json.loads(line);op=e.get('args',{}).get('op');kind={'gk_moe_gp_folded_v1':'gp','downact_direct_down_broadcast':'down'}.get(op)
    if not kind or e.get('ph') not in ('B','E'):continue
    key=(e['pid'],e['tid'],e['name'],e['id'])
    if e['ph']=='B':
        assert key not in starts;starts[key]=e
    else:
        begin=starts.pop(key);assert e['ts']>=begin['ts'];intervals[kind].append((begin['ts'],e['ts'],e['pid'],e['tid']));names[kind].add(e['name'])
assert not starts
result=dict(trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),device_acquired=False,
            scope='Actual rank0 native trace; overlap union across TPC cores, not clock or stall attribution',records={})
for kind,rows in intervals.items():
    groups=[]
    for row in sorted(rows):
        if not groups or row[0]>groups[-1]['end']:
            groups.append(dict(start=row[0],end=row[1],cores={(row[2],row[3])}))
        else:
            g=groups[-1];g['end']=max(g['end'],row[1]);g['cores'].add((row[2],row[3]))
    assert len(groups)==207, 'This input is the preserved 3-token/69-MoE-layer production70-c trace'
    for g in groups:g['duration_us']=g['end']-g['start'];g['cores']=sorted(g['cores']);g['core_count']=len(g['cores'])
    chunks=[]
    for i in range(3):
        chunk=groups[69*i:69*(i+1)];dur=[g['duration_us'] for g in chunk]
        chunks.append(dict(chronological_token_chunk=i,moe_nodes=69,union_us=sum(dur),median_node_us=statistics.median(dur),minimum_node_us=min(dur),maximum_node_us=max(dur),all_24_cores=all(g['core_count']==24 for g in chunk)))
    result['records'][kind]=dict(core_intervals=len(rows),node_envelopes=len(groups),chunks=chunks,names=sorted(names[kind]),envelopes=groups)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v['chunks'] for k,v in result['records'].items()}))
