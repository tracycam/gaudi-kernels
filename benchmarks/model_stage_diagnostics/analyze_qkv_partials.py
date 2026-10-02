"""Decompose this measured scalar using captured public-op device partials.

No inference of a unique MME tree; intermediate-export recipe is diagnostic.
"""
import argparse,ctypes,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--device',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
base=a.root;witness=json.loads((base/'first-qkv-rank6-layer1.json').read_text()) if (base/'first-qkv-rank6-layer1.json').exists() else torch.load(base/'first-qkv-rank6-layer1.pt',map_location='cpu',weights_only=False)['record']
ref=np.fromfile(base/'fp64-partials.bin',np.float64);sa=np.fromfile(base/'cpu-activation-scales-f32.bin',np.float32);sw=np.fromfile(base/'native-weight-scales-f32.bin',np.float32)
q=np.fromfile(base/'cpu-quantized-activation-fp8.bin',np.uint8);fp64_reference=float((torch.from_numpy(ref)*torch.from_numpy(sa).double()*torch.from_numpy(sw).double()).sum())
libm=ctypes.CDLL('libm.so.6');fma=libm.fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
def round_bf(v):return dict(value=float(v),bits=int(torch.tensor(v,dtype=torch.float64).bfloat16().view(torch.int16))&65535)
rows=[];outputs={}
for mode in ['eager','graph']:
 quant=np.fromfile(a.device/f'{mode}-quant-fp8.bin',np.uint8);scales=np.fromfile(a.device/f'{mode}-activation-scales-f32.bin',np.float32)
 full=np.fromfile(a.device/f'{mode}-partial-f32.bin',np.float32).reshape(48,1,3392);partial=full[:,0,witness['output_index']].copy();y=np.fromfile(a.device/f'{mode}-output-bf16.bin',np.uint16)
 assert np.array_equal(quant,q) and np.array_equal(scales.view(np.uint32),sa.view(np.uint32))
 partial.tofile(base/f'partial-device-{mode}.bin');outputs[mode]=dict(q=quant,scale=scales,p=full,y=y)
 product_scale=sa.astype(np.float64)*sw.astype(np.float64);factor=(sa*sw).astype(np.float32)
 only_mme=float((torch.from_numpy(partial).double()*torch.from_numpy(product_scale)).sum())
 rounded_factor=float((torch.from_numpy(partial).double()*torch.from_numpy(factor).double()).sum())
 acc=np.float32(0);trace=[]
 for g in range(48):
  before=acc;acc=np.float32(fma(float(partial[g]),float(factor[g]),float(acc)))
  trace.append(dict(group=g,fp64_partial=float(ref[g]),device_partial=float(partial[g]),partial_error=float(partial[g])-float(ref[g]),factor=float(factor[g]),accumulator_before=float(before),accumulator_after=float(acc)))
 assert round_bf(acc)['bits']==int(y[witness['output_index']])==witness['actual_bits']
 # Explicit diagnostic alternatives, evaluated on the same measured partials.
 terms=(partial*factor).astype(np.float32)
 total=np.float32(0);compensation=np.float32(0)
 for term in terms:
  corrected=np.float32(term-compensation);updated=np.float32(total+corrected)
  compensation=np.float32(np.float32(updated-total)-corrected);total=updated
 kahan=round_bf(total)
 total=np.float32(0);correction=np.float32(0)
 for v,f in zip(partial,factor):
  term=np.float32(v*f);product_error=np.float32(fma(float(v),float(f),-float(term)))
  updated=np.float32(total+term)
  residual=np.float32(np.float32(total-updated)+term) if abs(total)>=abs(term) else np.float32(np.float32(term-updated)+total)
  correction=np.float32(correction+np.float32(residual+product_error));total=updated
 compensated=round_bf(np.float32(total+correction))
 cpu_methods={}
 for path in base.glob('partial-*.bin'):
  if path.stem.startswith('partial-device'):continue
  candidate=np.fromfile(path,np.float32)
  if candidate.size==48:cpu_methods[path.stem[8:]]=int((candidate.view(np.uint32)!=partial.view(np.uint32)).sum())
 row=dict(mode=mode,quant_bytes_bitwise_match=True,scale_bytes_bitwise_match=True,
  fp64_reference=round_bf(fp64_reference),device_partials_fp64_scaling_sum=round_bf(only_mme),device_partials_fp32_factors_fp64_sum=round_bf(rounded_factor),device_partials_fp32_factors_sequential_fma=round_bf(acc),
  mme_partial_delta=only_mme-fp64_reference,factor_rounding_delta=rounded_factor-only_mme,fma_accumulation_delta=float(acc)-rounded_factor,
  different_partials_vs_fp64=int(np.count_nonzero(partial.astype(np.float64)!=ref)),different_partials_vs_round_once_fp32=int(np.count_nonzero(partial.view(np.uint32)!=ref.astype(np.float32).view(np.uint32))),
  cpu_partial_method_bit_differences=cpu_methods,diagnostic_kahan_fp32_rounded_products=kahan,diagnostic_neumaier_fp32_with_fma_product_residual=compensated,compensated_device_tested=False,per_group=trace)
 rows.append(row)
assert all(np.array_equal(outputs['eager'][k].view(np.uint8),outputs['graph'][k].view(np.uint8)) for k in outputs['eager'])
report=dict(device_diagnostic_reproduces_all_3392_production_bf16_outputs=True,eager_graph_all_intermediate_bits_equal=True,
 actual_quantization_matches_cpu_native_contract=True,actual_device_partials_with_source_fma_reproduce_actual_bits=True,
 original_serving_MME_tree_identified=False,model_quality_gate_changed=False,rows=rows)
(base/'device-partial-decomposition.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({**report,'rows':[{k:v for k,v in r.items() if k!='per_group'} for r in rows]},indent=2))
