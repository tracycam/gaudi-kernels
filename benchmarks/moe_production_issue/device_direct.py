"""Direct same-input FP32 diagnostic, exact failing fixture; no performance gate."""
import argparse,hashlib,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);a=p.parse_args()
assert os.environ.get('GAUDI_KERNELS_MODULE_ID')=='7' and os.environ.get('HABANA_VISIBLE_MODULES')=='7'
out=Path(os.environ['PROBE_OUT']);root=Path(__file__).resolve().parents[2]
import torch
import habana_frameworks.torch.core as hc
torch.set_num_threads(2)
for path in ('runtime/gaudi_moe_activation_folded.so','builds/torch/gaudi_gp_scale_tail.so'):torch.ops.load_library(str(root/path))
f=torch.load(a.fixture/'direct-gp-fp32.pt',map_location='cpu',weights_only=False)[0];w=torch.load(a.fixture/'prepared-weights.pt',map_location='cpu',weights_only=False)
values=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.bfloat16);table=torch.stack([values[torch.arange(256)%16],values[torch.arange(256)//16]],dim=1).reshape(-1)
gp,gs,x,ids,table=[t.to('hpu') for t in (w['gp'],w['gs'],f['x'],f['ids'],table)];hc.mark_step();torch.hpu.synchronize()
answers={}
for name,op in [('original',torch.ops.gaudi_activation_folded.gp),('diagnostic',torch.ops.gaudi_gp_scale_tail.gp)]:
 y=op(gp,gs,x,table,ids);hc.mark_step();torch.hpu.synchronize();answers[name]=y.cpu()
torch.save(answers,out/'outputs.pt');old,new=[answers[n].flatten() for n in ('original','diagnostic')];mask=old.view(torch.int32)!=new.view(torch.int32)
result=dict(device_used=True,checked=old.numel(),bit_mismatches=int(mask.sum()),candidate_nan=int(torch.isnan(new).sum()),old_finite=bool(torch.isfinite(old).all()),reference_unchanged=torch.equal(old,f['old'].flatten()),all_pass=not bool(mask.any()),scope='causal direct-GP diagnostic only, not full MoE or performance acceptance',first_bad_indices=mask.nonzero().flatten()[:16].tolist())
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
raise SystemExit(0 if result['all_pass'] and result['reference_unchanged'] else 1)
