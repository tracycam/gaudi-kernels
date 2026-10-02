"""Check emitted graph placement/byte intervals; does not infer physical HBM transactions."""
import argparse,json,math,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case-dir',type=Path,required=True);a=p.parse_args()
case=a.case_dir;records=[]
for line in (case/'run.log').read_text().splitlines():
 try:records.append(json.loads(line))
 except json.JSONDecodeError:pass
plan=next(r for r in records if r.get('stage')=='plan');assert plan['mode']=='mme'
graph_path=case/'post_graph.json'
if graph_path.is_dir():
 candidates=list(graph_path.glob('*.post.json'));assert len(candidates)==1
 graph_path=candidates[0]
graph=json.loads(graph_path.read_text())['graphs'][0]
tensors={t['name']:t for t in graph['tensors']};nodes=graph['nodes'];intervals={};decoded=[];mmes=[]
roots={name:t for name,t in tensors.items() if name.endswith(('_packed','_scales')) and t['persistent']}
for name,root in roots.items():intervals[name]=[]

def put(tensor,start,length):
 candidates=[(name,r) for name,r in roots.items() if r['dtype']==tensor['dtype'] and r['offset']<=start<r['offset']+math.prod(r['max_shape'])]
 assert len(candidates)==1,(tensor['name'],start,candidates)
 name,root=candidates[0];assert start+length<=root['offset']+math.prod(root['max_shape'])
 intervals[name].append((start-root['offset'],start-root['offset']+length))

previous_mme_consumers=[]
for node in nodes:
 if node.get('engine')=='MME':
  mmes.append({k:node.get(k) for k in ['name','mme_compute_utilization','mme_expected_compute_cycles','mme_node_strategy','input_tensors','output_tensors']})
 if not node['guid'].startswith('gk_mx4c3_'):continue
 w,s=[tensors[n] for n in node['input_tensors'][:2]];y=tensors[node['output_tensors'][0]]
 assert y['allocation']=='SRAM' and y['rmw_section'] and not y['persistent'],y
 assert w['alias'] and s['alias'] and w['allocation']=='DRAM' and s['allocation']=='DRAM'
 consumers=[n for n in nodes if y['name'] in n['input_tensors']]
 assert consumers and all(n['engine']=='MME' for n in consumers)
 if previous_mme_consumers:
  assert all(name in node['blocking_nodes'] for name in previous_mme_consumers),(node['name'],node['blocking_nodes'],previous_mme_consumers)
 previous_mme_consumers=[n['name'] for n in consumers]
 decoded.append({'node':node['name'],'tensor':y['name'],'shape':y['max_shape'],'allocation':y['allocation'],'offset':y['offset'],
                 'input_weight_view':w['name'],'input_scale_view':s['name'],'mme_consumers':[n['name'] for n in consumers],
                 'blocking_nodes':node['blocking_nodes']})
 if 'native' in node['guid']:
  block,pair0=struct.unpack('<ii',bytes(node['params']));n,k=y['max_shape'];assert w['strides'][0]==s['strides'][0]==1
  for pair in range(n//256):
   for g in range(k//32):
    for half in range(2):put(s,s['offset']+block*s['strides'][2]+g*s['strides'][1]+(pair+pair0)*256+half*128,128)
    for z in range(g*32,(g+1)*32):put(w,w['offset']+block*w['strides'][2]+z*w['strides'][1]+(pair+pair0)*128,128)
 else:
  k,n=y['max_shape'];assert w['strides'][0]==s['strides'][0]==1
  task=256 if 'rows256' in node['guid'] else 32
  for row in range(n):
   for g in range((k+task-1)//task):
    count=min(task,k-g*task);put(w,w['offset']+row*w['strides'][1]+g*(task//2),(count+1)//2)
    put(s,s['offset']+row*s['strides'][1]+g*(task//32),(count+31)//32)
ledger={}
for name,spans in intervals.items():
 cursor=0
 for start,stop in sorted(spans):assert start==cursor and stop>start,(name,cursor,start,stop);cursor=stop
 assert cursor==math.prod(roots[name]['max_shape'])
 ledger[name]={'unique_payload_bytes':cursor,'declared_requests':len(spans),'requested_elements_once':True}
assert sum(v['unique_payload_bytes'] for v in ledger.values())==plan['original_weight_scale_bytes']
assert len(decoded)==plan['decode_nodes'] and len(mmes)==plan['mme_nodes']
physical_other=[n for n in nodes if not n['is_logical'] and n['engine'] not in ['TPC','MME']]
result={'state':'compiled_graph_alias_sram_and_element_schedule_pass','plan':plan,'decoded':decoded,'weight_scale_ledger':ledger,
 'mme_nodes':mmes,'extra_physical_engines':physical_other,'physical_hbm_transactions':'not_measured',
 'scope':'Graph allocations plus ISA source request intervals. No repeated original weight elements in this graph schedule; cacheline/bus reads remain separate.'}
(case/'graph-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'state':result['state'],'case':case.name,'decoded_sram_tiles':len(decoded),'payload_bytes':sum(v['unique_payload_bytes'] for v in ledger.values())}))
