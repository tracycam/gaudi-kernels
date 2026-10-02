"""Read-only raw-bit old/fixed bridge comparison and physical metadata dtype audit."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--case-root',type=Path,required=True);p.add_argument('--prior-whole',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def tensors(value,prefix=''):
 if isinstance(value,torch.Tensor):yield prefix,value
 elif isinstance(value,dict):
  for k,v in value.items():yield from tensors(v,prefix+'.'+str(k))
 elif isinstance(value,list):
  for k,v in enumerate(value):yield from tensors(v,prefix+'.'+str(k))
records=[]
def compare(left,right,label):
 old=dict(tensors(torch.load(left,weights_only=False)));new=dict(tensors(torch.load(right,weights_only=False)));assert old.keys()==new.keys()
 for name,x in old.items():
  y=new[name];assert x.shape==y.shape and x.dtype==y.dtype
  bad=int((x.contiguous().view(torch.uint8)!=y.contiguous().view(torch.uint8)).sum());assert not bad,(label,name,bad)
  records.append(dict(pair=label,tensor=name,elements=x.numel(),dtype=str(x.dtype),mismatched_bytes=bad))
for state in ('initial','changed'):
 compare(a.case_root/'old-model512-a'/(state+'.pt'),a.case_root/'fixed-topk-model512-a'/(state+'.pt'),'model_ids_'+state)
for state in ('uniform','hot','skew','zero','invalid','restored'):
 compare(a.prior_whole/(state+'.pt'),a.case_root/'fixed-whole-synthetic-a'/(state+'.pt'),'whole_'+state)
physical=[];post=a.case_root/'fixed-whole-synthetic-a/post_graph.json'
for path in (post.rglob('*.json')if post.is_dir()else[post]):
 for graph in json.loads(path.read_text()).get('graphs',[]):
  byname={t['name']:t for t in graph['tensors']}
  for node in graph['nodes']:
   if node['guid']in('gk_route_count_i32_v3','gk_route_prefix_i32_v3','gk_route_inverse_i32_v3','gk_route_row_map_i32_v3'):
    dtypes={name:byname[name]['dtype']for name in node['input_tensors']+node['output_tensors']}
    assert all(v=='int32'for v in dtypes.values()),dtypes
    physical.append(dict(graph=graph['name'],guid=node['guid'],dtypes=dtypes))
assert len(physical)==4
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(status='PASS_OLD_FIXED_RAW_BITS_PHYSICAL_I32',comparisons=records,physical_metadata_nodes=physical,
 scope='Independent old/fixed saved bytes and physical I32 node metadata; not a model output acceptance or proof that this synthetic graph exercised logical Long.'),indent=2)+'\n')
print(json.dumps(dict(status='PASS',comparisons=len(records),elements=sum(r['elements']for r in records))))
