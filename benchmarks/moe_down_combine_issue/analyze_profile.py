"""Last ABBA replay phase: physical down and whole recipe, separate from host spans."""
import argparse,collections,csv,hashlib,json,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
result=json.loads(a.result.read_text());assert result['all_numeric_pass'] and result['placement']['pass_owner_and_descriptor_gate'] and result['event_wall_consistent']
raw_rows=list(csv.DictReader(a.csv.open()))
# Synapse's CSV may repeat the same physical fused node once for each
# represented source node. Collapse only identical physical interval records.
physical={}
for row in raw_rows:
    key=tuple(row[k] for k in ('Graph Name','Unique node id','Start time of node','End time of node'))
    if key in physical:
        previous=physical[key]
        assert all(previous[k]==row[k] for k in ('Node Name','Op Type','Duration (us)','Parallel Engines'))
    else:physical[key]=row
rows=list(physical.values());guids={'baseline':'downact_direct_down_broadcast','p4':'gk_down_combine_m1_p4_v0','p6':'gk_down_combine_m1_p6_v0'}
allnodes=sorted([r for r in rows if r['Op Type'] in guids.values()],key=lambda r:float(r['Start time of node']))
required=sum(r['replays'] for r in result['timing']);assert required>0;selected=allnodes[-required:];offset=0;segments=[];chosen={}
for i,record in enumerate(result['timing']):
    part=selected[offset:offset+record['replays']];offset+=record['replays'];assert len(part)==record['replays']
    assert all(r['Op Type']==guids[record['variant']] and int(r['Parallel Engines'])==24 for r in part)
    segments.append(dict(index=i,candidate=record['candidate'],variant=record['variant'],kernel_median_us=statistics.median(float(r['Duration (us)']) for r in part),replays=len(part)))
    for r in part:chosen[(r['Graph Name'],r['Start time of node'])]=(record['candidate'],record['variant'])
assert len(chosen)==required
samples=collections.defaultdict(list)
for graph in {r['Graph Name'] for r in selected}:
    nodes=sorted([r for r in rows if r['Graph Name']==graph],key=lambda r:float(r['Start time of node']))
    names=collections.Counter(re.sub(r' iter \d+$','',r['Node Name']) for r in nodes)
    assert len(set(names.values()))==1,('partial trace or variable physical node set',graph)
    count=len(names)
    for i in range(0,len(nodes),count):
        chunk=nodes[i:i+count];assert {re.sub(r' iter \d+$','',r['Node Name']) for r in chunk}==set(names)
        down=[r for r in chunk if r['Op Type'] in guids.values()];assert len(down)==1;down=down[0]
        key=(graph,down['Start time of node'])
        if key not in chosen:continue
        combine=[r for r in chunk if r['Op Type']=='pf_combine_f32']
        tail=combine[0] if combine else down
        samples[chosen[key]].append(dict(kernel_us=float(down['Duration (us)']),down_through_combine_us=float(tail['End time of node'])-float(down['Start time of node']),whole_recipe_us=max(float(r['End time of node']) for r in chunk)-min(float(r['Start time of node']) for r in chunk)))
output=dict(raw_csv_rows=len(raw_rows),physical_rows=len(rows),duplicate_physical_rows=len(raw_rows)-len(rows),segments=segments,pairs={},physical_HBM_measured=False,clock_measured=False,model_gain_measured=False,csv_sha256=hashlib.sha256(a.csv.read_bytes()).hexdigest(),scope='Actual node and recipe envelopes in final ABBA phase; no host event rescaling')
for (candidate,variant),values in samples.items():
    want=sum(r['replays'] for r in result['timing'] if r['candidate']==candidate and r['variant']==variant);assert len(values)==want
    output['pairs'].setdefault(candidate,{})[variant]=dict(samples=want,medians={k:statistics.median(r[k] for r in values) for k in values[0]})
a.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output['pairs']))
