"""Independent canonical-byte CPU FP32 MoE checks of saved device outputs.

Different FP32 trees are reported, not required to match an FP64 oracle. This
is sampled operator correctness evidence, not a whole-model quality decision.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--states',nargs='+',default=['checkpoint','hot8','mixed']);a=p.parse_args()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.mxfp4_reference import MXFP4Weights,moe_reference
from gaudi_kernels.serving.executor.packing import unpack
assert hashlib.sha256(a.fixture.read_bytes()).hexdigest()=='6ebc2f3734fce93bf05c05fd46f8b94997f600653ab1f6860710349aa014dcca'
torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
fixture=torch.load(a.fixture,map_location='cpu',weights_only=False);report=[]
for state in a.states:
 files=sorted(a.results.glob(state+'-*.pt'));assert files,state
 sample=torch.load(files[0],map_location='cpu',weights_only=False);inputs=sample['inputs'];t=inputs['x'].shape[0];rows=sorted(set([0,t//2,t-1]));ids=inputs['ids'][rows].long();experts=ids.unique(sorted=True);local=ids.clone()
 owners={}
 for name,scale,n,k in [('gp','gs',512,6144),('dp','ds',6144,256)]:
  ws=[];ss=[]
  for i,e in enumerate(experts.tolist()):
   w=fixture[name][e*(n//512)*k:(e+1)*(n//512)*k].reshape(n//512,k,256).numpy()
   s=fixture[scale][e*(n//512)*(k//32):(e+1)*(n//512)*(k//32)].reshape(n//512,k//32,512).numpy()
   w,s=unpack(w,s);ws.append(torch.from_numpy(w));ss.append(torch.from_numpy(s))
   local[ids==e]=i
  owners[name]=MXFP4Weights(torch.stack(ws),torch.stack(ss),k)
 ref=moe_reference(inputs['x'][rows],local,inputs['routing'][rows],owners['gp'],owners['dp'],policy='w4a16')
 torch.save(dict(reference=ref,rows=rows,expert_ids=experts),a.output/(state+'-reference.pt'))
 for f in files:
  value=torch.load(f,map_location='cpu',weights_only=False)['y'][rows];delta=value-ref
  assert torch.isfinite(value).all()
  report.append(dict(file=f.name,rows=rows,relative_l2=float(delta.norm()/ref.norm().clamp_min(1e-30)),max_abs=float(delta.abs().max()),reference_norm=float(ref.norm())))
(a.output/'result.json').write_text(json.dumps(dict(scope='sampled canonical MXFP4, CPU FP32 K32 MoE reference; precision policy deferred',checks=report),indent=2)+'\n')
print(json.dumps(report,indent=2))
