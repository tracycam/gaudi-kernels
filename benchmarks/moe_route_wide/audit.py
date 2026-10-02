"""Read actual C32 or wide PostGraphs; no device APIs or traffic inference."""
import argparse,collections,hashlib,json,math,struct
from pathlib import Path
DECODE='gk_mxfp4_graph_decode_historical'
OLD_META=tuple('gk_route_'+s+'_i32_v3'for s in('count','prefix','inverse','row_map'))
WIDE_META=tuple('gk_route_wide_'+s+'_i32_v3'for s in('count','prefix','inverse','row_map'))

def span(t):
 shape=t['max_shape'];strides=t['strides'];bits=t['dtype_bit_size']
 assert bits>0 and bits%8==0 and len(strides)>=len(shape) and all(x>0 for x in shape) and all(x>=0 for x in strides)
 return bits//8+sum((n-1)*stride for n,stride in zip(shape,strides))
def union(regions):
 total=0;end=None
 for begin,size in sorted(regions):
  hi=begin+size
  if end is None or begin>=end:total+=size;end=hi
  elif hi>end:total+=hi-end;end=hi
 return total

def audit_graph(graph,source):
 nodes=graph['nodes'];decoders=[n for n in nodes if n['guid']==DECODE]
 if not decoders:return None
 tensors={t['name']:t for t in graph['tensors']};counts=collections.Counter(n['guid']for n in nodes);issues=[];consumers=collections.defaultdict(list);aliases=collections.defaultdict(list)
 for n in nodes:
  for name in n['input_tensors']:consumers[name].append(n)
 for t in tensors.values():
  if t.get('alias_of'):aliases[t['alias_of']].append(t['name'])
 wide=any(counts[n]for n in WIDE_META);meta=WIDE_META if wide else OLD_META;other=OLD_META if wide else WIDE_META
 if any(counts[n]!=1 for n in meta)or any(counts[n]for n in other):issues.append('metadata producer must appear exactly once, with no mixed namespaces')
 prefix=[n for n in nodes if n['guid']==meta[1]];tile_rows=[];capacities=[]
 for n in prefix:
  if len(n.get('params',[]))!=12:issues.append('invalid prefix params');continue
  routes,c,b=struct.unpack('<iii',bytes(n['params']));tile_rows.append(c);capacities.append(b)
  if not 1<=routes<=8 or not 1<=c<=(128 if wide else 32)or b<1:issues.append('prefix geometry outside declared namespace')
 consumers_prefix='gk_route_wide_tile_'if wide else'gk_route_tile_'
 for kind in('gather_bf16_v1','gate_bf16_v1','combine_f32_v1'):
  if not counts[consumers_prefix+kind]:issues.append('missing route consumer '+kind)
 gather_c=set()
 for n in nodes:
  if n['guid']in meta:
   for name in n['input_tensors']+n['output_tensors']:
    if tensors[name]['dtype']!='int32':issues.append('metadata physical dtype not int32: '+name)
  if n['guid']==consumers_prefix+'gather_bf16_v1':
   if len(n.get('params',[]))!=12:issues.append('invalid gather params');continue
   gather_c.add(struct.unpack('<iii',bytes(n['params']))[1])
 if len(tile_rows)!=1 or gather_c!=set(tile_rows):issues.append('prefix/gather logical C disagreement')
 seen=set();reached_mme=set();decoded=[]
 def walk(name):
  if name in seen:return
  seen.add(name)
  if name not in tensors:issues.append('missing expanded tensor '+name);return
  t=tensors[name];decoded.append({k:t.get(k)for k in('name','max_shape','dtype','allocation','persistent','offset','alias','alias_of','user_mem_section_index')})
  if t.get('allocation')!='SRAM'or t.get('persistent'):issues.append('expanded weight not transient SRAM: '+name)
  if t.get('dtype')!='bf16':issues.append('expanded weight precision changed: '+name)
  for alias in aliases[name]:walk(alias)
  for n in consumers[name]:
   if n['guid']in('batch_gemm','gemm'):
    if len(n['input_tensors'])<2 or n['input_tensors'][1]!=name:issues.append('expanded tensor is not MME B: '+name)
    else:reached_mme.add(n['name'])
   elif n.get('is_logical')or n.get('engine')=='DMA':
    for output in n['output_tensors']:walk(output)
   else:issues.append('unreviewed expanded-weight consumer '+n['guid']+': '+name)
 owners={};schedule=[]
 for n in decoders:
  if len(n.get('params',[]))!=12:issues.append('invalid decoder params');continue
  nblocks,slot,nb=struct.unpack('<iii',bytes(n['params']))
  for name in n['input_tensors'][:2]:
   t=tensors[name];owners[name]={k:t.get(k)for k in('max_shape','dtype','allocation','persistent','alias','alias_of')}
   if t.get('dtype')!='uint8'or t.get('allocation')!='DRAM'or not t.get('persistent'):issues.append('original-width persistent DRAM owner changed: '+name)
  for name in n['output_tensors']:walk(name)
  schedule.append(dict(node=n['name'],nblocks=nblocks,slot_begin=slot,nblock_begin=nb,output_shapes=[tensors[x]['max_shape']for x in n['output_tensors']],weight_owner=n['input_tensors'][0],scale_owner=n['input_tensors'][1]))
 mme=[]
 for n in nodes:
  if n['guid']not in('batch_gemm','gemm'):continue
  if n['name']not in reached_mme:issues.append('MME B not reached from transient decoder: '+n['name'])
  ins=[tensors[x]for x in n['input_tensors']];outs=[tensors[x]for x in n['output_tensors']]
  if len(ins)<2 or any(t['dtype']!='bf16'for t in ins[:2])or any(t['dtype']!='float32'for t in outs):issues.append('BF16-input/FP32-output MME contract changed: '+n['name'])
  mme.append(dict(name=n['name'],guid=n['guid'],A=ins[0]['max_shape'],B=ins[1]['max_shape'],outputs=[t['max_shape']for t in outs],strategy=n.get('mme_node_strategy'),compiler_expected_cycles=n.get('mme_expected_compute_cycles'),compiler_compute_utilization=n.get('mme_compute_utilization'),decoded_B=bool(n['name']in reached_mme)))
 if not mme:issues.append('no MME consumer')
 dma=[]
 for n in nodes:
  if n.get('engine')!='DMA':continue
  ins=[tensors[x]for x in n['input_tensors']];outs=[tensors[x]for x in n['output_tensors']]
  dma.append(dict(name=n['name'],guid=n['guid'],inputs=[{k:t.get(k)for k in('name','max_shape','dtype','allocation','persistent','alias_of')}for t in ins],outputs=[{k:t.get(k)for k in('name','max_shape','dtype','allocation','persistent')}for t in outs],logical_output_bytes=sum(math.prod(t['max_shape'])*t['dtype_bit_size']//8 for t in outs),output_roles=[dict(consumer=c['name'],guid=c['guid'],operand=c['input_tensors'].index(t['name']))for t in outs for c in consumers[t['name']]]))
 def regions(names):return[(int(tensors[name]['offset']),span(tensors[name]))for name in names if tensors[name].get('allocation')=='SRAM']
 return dict(graph=graph['name'],source=source,namespace='wide'if wide else'original_v3',logical_C=tile_rows,capacity=capacities,decoder_nodes=len(decoders),compute_nodes=sum(not n['is_logical']for n in nodes),MME_nodes=len(mme),guids=dict(counts),workspace_bytes=graph['workspace_size'],decoded_weight_SRAM_address_union_bytes=union(regions(seen)),all_SRAM_address_union_bytes=union(regions(tensors)),original_owners=owners,decode_schedule=schedule,decoded=decoded,MME=mme,DMA=dma,placement_pass=not issues,issues=issues,physical_HBM_measured=False,limitations='Node counts are compiled nodes, not host launches. SRAM union is address union, not simultaneous lifetime proof or physical traffic. Compiler MME cycle/utilization fields are estimates, not measured engine time.')

def main():
 p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();post=a.case/'post_graph.json';files=sorted(post.rglob('*.json'))if post.is_dir()else[post];records=[];sources={}
 for f in files:
  data=f.read_bytes();sources[str(f)]=hashlib.sha256(data).hexdigest()
  for graph in json.loads(data).get('graphs',[]):
   r=audit_graph(graph,str(f))
   if r:records.append(r)
 report=dict(all_placement_pass=bool(records)and all(r['placement_pass']for r in records),records=records,source_sha256=sources)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(all_placement_pass=report['all_placement_pass'],graphs=len(records),C=[r['logical_C']for r in records])))
 if not report['all_placement_pass']:raise SystemExit(1)
if __name__=='__main__':main()
