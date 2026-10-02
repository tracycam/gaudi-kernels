"""CPU model plus extracted pinned plugin arithmetic. No HPU claim."""
import argparse,ast,hashlib,json,math,types
from pathlib import Path
import torch
from oracle import reference,requested_bytes,visible_slots
p=argparse.ArgumentParser();p.add_argument('--plugin-ops',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
source=a.plugin_ops.read_text();names={'grouped_max','b2b_impl','batch2block','block2batch','pipelined_pa'};tree=ast.parse(source);extracted=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
namespace={'torch':torch,'math':math,'get_config':lambda:types.SimpleNamespace(fused_block_softmax=False,fused_block_softmax_adjustment=False)};exec(compile(extracted,str(a.plugin_ops),'exec'),namespace)
(a.output/'plugin-functions.py').write_text(ast.unparse(extracted)+'\n');torch.set_num_threads(2);torch.manual_seed(281715)
q=torch.randn(1,16,192).bfloat16();k=torch.randn(512,1,192).bfloat16();v=torch.randn(512,1,128).bfloat16();records=[]
for pos in [0,64,127,128,192,255]:
 for sink_mode in ['none','finite']:
  starts=[(pos//128-1)*128,(pos//128)*128];ids=[3 if starts[0]>=0 else -1,1];sink=torch.full((16,),-torch.inf,dtype=torch.bfloat16) if sink_mode=='none' else torch.randn(16).bfloat16()
  expected,stages=reference(q,k,v,ids,starts,pos,sink);slots=visible_slots(ids,starts,pos,len(k));values=torch.zeros((2,1,128,128),dtype=torch.bfloat16)
  for page,rows in enumerate(slots):
   for t,slot in rows:values[page,0,t]=v[slot,0]
  # QK uses the independent complete FP64 MAC then BF16 score boundary. The
  # extracted plugin functions audit subsequent arithmetic/layout unchanged.
  scores=torch.stack(stages['logits'])[:,:,None,:];groups=torch.zeros(2,dtype=torch.long);mapping=torch.ones((2,1),dtype=torch.bfloat16);sinks=sink[None,:,None,None].expand(2,-1,-1,-1)
  partial=namespace['pipelined_pa'](scores,values,None,groups,mapping,sinks,1,torch.matmul,torch.matmul,torch.matmul)
  actual=namespace['block2batch'](partial,mapping).reshape(1,16,128)
  rel=float((actual.double()-expected.double()).norm()/expected.double().norm().clamp_min(1e-30));assert torch.isfinite(actual).all() and rel<.003,(pos,sink_mode,rel)
  name=f'p{pos}-{sink_mode}';torch.save({'q':q,'k':k,'v':v,'ids':ids,'starts':starts,'position':pos,'sinks':sink,'oracle':expected,'plugin_cpu':actual,'stages':stages},a.output/(name+'.pt'))
  records.append({'case':name,'plugin_cpu_relative_l2':rel,'bitwise_equal':torch.equal(actual,expected),'requested_bytes':requested_bytes(ids,starts,pos,len(k))})
for sink in [torch.zeros(16,dtype=torch.bfloat16),torch.full((16,),-torch.inf,dtype=torch.bfloat16)]:
 y,stages=reference(q,k,v,[-1,-1],[-128,0],0,sink);assert torch.equal(y,torch.zeros_like(y));assert all(not p.any() for p in stages['probabilities'])
for bad in [([1,2],[0,0],1),([4,-1],[0,128],1),([1,2],[0,129],1)]:
 try:visible_slots(*bad,len(k))
 except ValueError:pass
 else:raise AssertionError('invalid descriptor accepted')
result={'status':'PASS_CPU_ARITHMETIC_AND_PAGE_GATES','device_verified':False,'vendor_hpu_numerics_verified':False,'plugin_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'records':records,'all_masked_sink_and_no_sink_zero':True,'checks':['offset0_64_127','two_pages','single_sink_after_page_aggregation','invalid_page_no_reads','duplicate_or_unaligned_logical_page_rejected','full_fp64_QK_AV_reference'],'scope':'extracted plugin post-QK functions run on CPU; not HPU exp/div/reduction equivalence'};(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'cases':len(records),'max_relative_l2':max(r['plugin_cpu_relative_l2'] for r in records)}))
