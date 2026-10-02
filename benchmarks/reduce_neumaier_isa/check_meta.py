"""Positive/negative schemas plus coexistence with the old reduction namespace."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--old-library',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
for path in [a.old_library,a.library]:torch.ops.load_library(str(path.resolve()))
checks=[]
for g,m,n in [(48,1,3392),(48,16,3392),(1,16,128),(5,3,257)]:
 partial=torch.empty(g,m,n,device='meta');sa=torch.empty(g,m,1,device='meta');sw=torch.empty((n+127)//128,g,device='meta');bias=torch.empty(n,device='meta')
 old=torch.ops.gk_reduce_experiment.neumaier(partial,sa,sw,bias)
 for name in ['baseline','lookahead','handschedule','baseline_debug','lookahead_debug','handschedule_debug']:
  op=getattr(torch.ops.gk_reduce_isa,name);y=op(partial,sa,sw,bias)
  assert y.shape==old.shape and y.dtype==(torch.float32 if name.endswith('_debug') else old.dtype)
  checks.append(dict(groups=g,rows=m,columns=n,op=name,pass_meta=True))
 for invalid in [(partial.half(),sa,sw,bias),(partial,sa[:,:,:0],sw,bias),(partial,sa,sw[:,:0],bias),(partial,sa,sw,bias[:n-1]),(partial.clone().requires_grad_(),sa,sw,bias)]:
  try:torch.ops.gk_reduce_isa.handschedule(*invalid)
  except RuntimeError:checks.append(dict(rejected_invalid=True))
  else:raise AssertionError('invalid input accepted')
a.out.write_text(json.dumps(dict(all_pass=True,device_tested=False,old_and_new_namespaces_coexist=True,checks=checks),indent=2)+'\n')
