"""Audit actual route-tile graph placement and logical original-width reads."""
import argparse,collections,json,math,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
post=a.case/'post_graph.json';files=sorted(post.rglob('*.json')) if post.is_dir() else [post]
records=[]
def byte_span(t):
 return t['dtype_bit_size']//8+sum((n-1)*stride for n,stride in zip(t['max_shape'],t['strides']))
def union_bytes(intervals):
 total=0;end=None
 for begin,size in sorted(intervals):
  hi=begin+size
  if end is None or begin>=end:total+=size;end=hi
  elif hi>end:total+=hi-end;end=hi
 return total

for path in files:
 for graph in json.loads(path.read_text()).get('graphs',[]):
  nodes=graph['nodes'];dec=[n for n in nodes if n['guid']=='gk_mxfp4_graph_decode_historical']
  if not dec:continue
  tensors={t['name']:t for t in graph['tensors']};decoded=[];seen=set();bad=[]
  def check_tensor(name):
   if name in seen:return
   seen.add(name);t=tensors[name];decoded.append({k:t.get(k) for k in ['name','max_shape','dtype','allocation','persistent','rmw_section','user_mem_section_index','offset','alias','alias_of']})
   if t.get('allocation')!='SRAM' or t.get('persistent'):bad.append('expanded weight not transient SRAM: '+name)
   for alias in tensors.values():
    if alias.get('alias_of')==name:check_tensor(alias['name'])
   for consumer in nodes:
    if name not in consumer['input_tensors']:continue
    if consumer['guid'] in ('batch_gemm','gemm'):
     if consumer['input_tensors'][1]!=name:bad.append('decode is not MME B: '+name)
    else:
     for output in consumer['output_tensors']:check_tensor(output)
  for d in dec:
   for name in d['output_tensors']:check_tensor(name)
  weight_regions=[(int(tensors[n]['offset']),byte_span(tensors[n])) for n in seen if tensors[n].get('allocation')=='SRAM']
  all_sram_regions=[(int(t['offset']),byte_span(t)) for t in tensors.values() if t.get('allocation')=='SRAM']
  original_owners={};schedules=[]
  for d in dec:
   par=struct.unpack('<iii',bytes(d['params']));y=tensors[d['output_tensors'][0]];N,K,B=y['max_shape'][:3]
   schedules.append(dict(nblocks=par[0],slot_begin=par[1],nblock_begin=par[2],local_experts=B,N=N,K=K,weight_owner=d['input_tensors'][0],scale_owner=d['input_tensors'][1]))
   for name in d['input_tensors'][:2]:
    t=tensors[name];original_owners[name]={k:t.get(k) for k in ['max_shape','dtype','allocation','persistent','alias','alias_of']}
    if t.get('dtype')!='uint8' or t.get('allocation')!='DRAM' or not t.get('persistent'):bad.append('original owner changed: '+name)
  counts=collections.Counter(n['guid'] for n in nodes)
  metadata_version=3 if counts['gk_route_count_i32_v3'] else 2
  metadata_guids=('gk_route_count_i32_v3','gk_route_prefix_i32_v3','gk_route_inverse_i32_v3','gk_route_row_map_i32_v3') if metadata_version==3 else ('gk_route_count_i32_v2','gk_route_prefix_i32_v2','gk_route_scatter_i32_v2')
  if any(counts[name]!=1 for name in metadata_guids):bad.append('metadata producer not exactly once in compiled graph')
  records.append(dict(graph=graph['name'],decoder_nodes=len(dec),compute_nodes=sum(not n['is_logical'] for n in nodes),guids=dict(counts),workspace_bytes=graph['workspace_size'],decoded_weight_SRAM_address_union_bytes=union_bytes(weight_regions),all_SRAM_address_union_bytes=union_bytes(all_sram_regions),original_owners=original_owners,decode_schedule=schedules,decoded=decoded,placement_pass=not bad,issues=bad,physical_HBM_measured=False,logical_weight_once_requires_dynamic_map_plus_ISA_proof=True,route_tile_reuse_scope='Original-width weights may be reloaded for further C-row tiles of one expert; this is not a claim of once per whole batch'))
result=dict(records=records,all_placement_pass=bool(records) and all(r['placement_pass'] for r in records),device_numeric_result=json.loads((a.case/'result.json').read_text()) if (a.case/'result.json').exists() else None)
(a.output or a.case/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(all_placement_pass=result['all_placement_pass'],graphs=len(records))))
