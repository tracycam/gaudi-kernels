"""Per-recipe engine/node interval unions; overlapping families are not additive."""
import argparse,collections,csv,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def union(spans):
 total=0.;end=float('-inf')
 for lo,hi in sorted(spans):total+=max(0.,hi-max(lo,end));end=max(end,hi)
 return total
buckets=collections.defaultdict(list)
for row in csv.DictReader(a.csv.open()):
 it=re.search(r' iter (\d+)',row['Node Name'])
 buckets[row['Graph Name'],int(it[1]) if it else 1].append(row)
recipes=[]
for (graph,iteration),rows in sorted(buckets.items()):
 families=collections.defaultdict(list);engines=collections.defaultdict(list);spans=[]
 for row in rows:
  span=(float(row['Start time of node']),float(row['End time of node']));spans.append(span);families[row['Op Type']].append(span);engines[row['Unit']].append(span)
 span=max(v[1] for v in spans)-min(v[0] for v in spans)
 recipes.append(dict(graph=graph,iteration=iteration,node_count=len(rows),span_us=span,all_nodes_union_us=union(spans),gap_us=span-union(spans),engine_unions_us={k:union(v) for k,v in engines.items()},families=sorted([dict(op=k,nodes=len(v),union_us=union(v),sum_us=sum(b-a for a,b in v)) for k,v in families.items()],key=lambda r:-r['union_us'])))
placement=[]
for graph in json.loads(a.graph.read_text())['graphs']:
 tensors={x['name']:x for x in graph['tensors']};nodes=[n for n in graph['nodes'] if n.get('guid') in ('gk_expert_decode_k8','gk_mxfp4_graph_decode_historical')]
 outputs=[tensors[name] for n in nodes for name in n['output_tensors']]
 placement.append(dict(graph=graph['name'],nodes=len(graph['nodes']),decode_nodes=len(nodes),sram_decode_outputs=sum(t['allocation']=='SRAM' for t in outputs),all_decoded_fragments_sram=all(t['allocation']=='SRAM' for t in outputs),workspace_bytes=graph['workspace_size']))
a.output.write_text(json.dumps(dict(scope='Profile attribution only; family/engine unions overlap and must not be summed. Compiled SRAM placement does not establish physical HBM traffic.',recipes=recipes,placement=placement),indent=2)+'\n')
print(len(recipes),'recipe iterations',len(placement),'compiled graphs')
