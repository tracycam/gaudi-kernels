"""Conditional CPU arithmetic reconstruction of the first captured QKV mismatch.

Actual MME partials are not observed. Plausible FP32 trees are controls, not a
claim about MME internals. The all-orders integer bound is checked separately.
"""
import argparse,ctypes,json,math
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--witness',type=Path,required=True);p.add_argument('--row',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
d=torch.load(a.witness,map_location='cpu',weights_only=False);x=d['input_bf16'].float().reshape(48,128)
w=torch.frombuffer(bytearray((a.row/'weight-e4m3-row.bin').read_bytes()),dtype=torch.uint8).view(torch.float8_e4m3fn).float().reshape(48,128)
sc=torch.frombuffer(bytearray((a.row/'scales-f32-row.bin').read_bytes()),dtype=torch.float32)
sa=x.abs().amax(-1,keepdim=True).clamp_min(1e-10)*(1./448.)
qa=(x/sa).clamp(-448,448).to(torch.float8_e4m3fn).float()
qa=(qa*.5).to(torch.float8_e4m3fn).float();w=(w*.5).to(torch.float8_e4m3fn).float();sa=sa[:,0]*2;sc=sc*2
products=qa.double()*w.double();partials=products.sum(-1);ref=float((partials*sa.double()*sc.double()).sum())
libm=ctypes.CDLL('libm.so.6');fma=libm.fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
factors=np.asarray(sa*sc,dtype=np.float32)
def bf(value):return dict(value=float(value),bf16=float(torch.tensor(value,dtype=torch.float64).bfloat16()),bits=int(torch.tensor(value,dtype=torch.float64).bfloat16().view(torch.int16))&65535)
def sequential(v):
 out=np.float32(0)
 for value in v:out=np.float32(out+np.float32(value))
 return out
def pairtree(v):
 v=np.asarray(v,dtype=np.float32)
 while len(v)>1:v=np.asarray(v[::2]+v[1::2],dtype=np.float32)
 return v[0]
def stripes(v,count):
 chains=[sequential(v[i::count]) for i in range(count)]
 return pairtree(chains)
def reduce(partial,fused):
 acc=np.float32(0)
 for v,f in zip(partial,factors):acc=np.float32(fma(float(v),float(f),float(acc))) if fused else np.float32(acc+np.float32(np.float32(v)*f))
 return float(acc)
# All products are dyadic. If sum(abs(integer coefficients)) < 2^24 after
# factoring out their shared power of two, every possible FP32 partial is exact.
bounds=[]
for group,values in enumerate(products.tolist()):
 ratios=[float(v).as_integer_ratio() for v in values if v]
 unit=min((-(den.bit_length()-1)+(abs(num)&-abs(num)).bit_length()-1 for num,den in ratios),default=0)
 coeff=[round(math.ldexp(v,-unit)) for v in values];bound=sum(abs(c) for c in coeff)
 bounds.append(dict(group=group,unit_exponent=unit,sum_abs_coefficients=bound,all_orders_fp32_exact=bound<=2**24,fp64_partial=float(partials[group]),fp64_partial_fp32_exact=float(np.float32(partials[group]))==float(partials[group])))
methods={'fp64_partials_cast_f32':partials.numpy().astype(np.float32)}
for label,fun in [('sequential',sequential),('balanced',pairtree),('stripe4',lambda v:stripes(v,4)),('stripe8',lambda v:stripes(v,8)),('stripe16',lambda v:stripes(v,16))]:methods[label]=np.asarray([fun(v) for v in products.numpy()],dtype=np.float32)
trials=[]
for name,partial in methods.items():
 partial.tofile(a.row/('partial-'+name+'.bin'))
 for fused in [True,False]:trials.append(dict(partial_method=name,reduction='FMA' if fused else 'mul_then_add',partial_difference_count=int(np.count_nonzero(partial.astype(np.float64)!=partials.numpy())),**bf(reduce(partial,fused))))
low=d['record']['reference_value'];high=d['record']['actual_value'];mid=(low+high)/2
report=dict(witness=d['record'],fp64_recomputed=bf(ref),bf16_midpoint=mid,fp64_minus_midpoint=ref-mid,midpoint_distance_in_fp32_ulps=(ref-mid)/float(np.spacing(np.float32(mid))),all_groups_all_orders_exact=all(r['all_orders_fp32_exact'] for r in bounds),bounds=bounds,trials=trials,actual_mme_partials_observed=False,actual_activation_quant_bytes_observed=False,device_accessed=False)
assert bf(ref)['bits']==d['record']['reference_bits'],'Original checkpoint row did not reproduce saved CPU oracle'
d['input_bf16'].view(torch.uint16).numpy().tofile(a.row/'input-bf16.bin');qa.to(torch.float8_e4m3fn).view(torch.uint8).numpy().tofile(a.row/'cpu-quantized-activation-fp8.bin')
np.asarray(qa,dtype=np.float32).tofile(a.row/'cpu-quantized-activation-f32.bin');np.asarray(sa,dtype=np.float32).tofile(a.row/'cpu-activation-scales-f32.bin');np.asarray(sc,dtype=np.float32).tofile(a.row/'native-weight-scales-f32.bin');partials.numpy().tofile(a.row/'fp64-partials.bin')
(a.row/'cpu-reconstruction.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='bounds'},indent=2))
