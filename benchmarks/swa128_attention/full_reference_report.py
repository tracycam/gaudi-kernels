"""Re-evaluate saved full outputs against FP64 attention with final rounding only."""
import argparse,json
from pathlib import Path
import torch
from oracle import full_reference
p=argparse.ArgumentParser();p.add_argument('device_case',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);rows=[]
for path in sorted(a.device_case.glob('*/raw.pt')):
 j=torch.load(path,weights_only=False);refs={};record={'case':path.parent.name,'variants':{}}
 for changed in [False,True]:
  q=(-j['query'] if changed else j['query']).reshape(1,16,192);high,rounded=full_reference(q,j['key'],j['value'],j['pages'],j['starts'],j['position'],j['sinks']);label='changed' if changed else 'initial';refs[label]={'fp64_unrounded':high,'final_bf16':rounded}
  for variant in ['vendor','candidate']:
   actual=j[('changed_' if changed else '')+variant];d=actual.double()-high;dr=actual.double()-rounded.double();record['variants'].setdefault(variant,{})[label]={'to_fp64_relative_l2':float(d.norm()/high.norm().clamp_min(1e-30)),'to_fp64_max_abs':float(d.abs().max()),'to_fp64_rms':float(d.square().mean().sqrt()),'to_final_bf16_relative_l2':float(dr.norm()/rounded.double().norm().clamp_min(1e-30)),'to_final_bf16_bit_mismatches':int((actual.view(torch.int16)!=rounded.view(torch.int16)).sum())}
 torch.save(refs,a.output/(path.parent.name+'.pt'));rows.append(record)
result={'contract':'BF16 Q/K/V/sink inputs; full FP64 QK, real scale192^-0.5, stable FP64 softmax/AV, final BF16 only; sink once; all-masked/no-sink zero','source_device_case':str(a.device_case),'records':rows};(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({v:max(d['variants'][v][k]['to_fp64_relative_l2'] for d in rows for k in ['initial','changed']) for v in ['vendor','candidate']}))
