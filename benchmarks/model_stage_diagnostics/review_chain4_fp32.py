#!/usr/bin/env python3
"""Reassess a saved cancellation counterexample; never rewrite old gate results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from gaudi_kernels.block_fp8_fp32_contract import build_reference, epilogue_certificate, require_fast_backend, _fma, _add, _gamma, _bf16

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
backend=require_fast_backend()

def chain4(partial,factors,bias):
 sums=[np.zeros(partial.shape[1:],np.float32) for _ in range(4)]
 for g in range(partial.shape[0]):sums[g%4]=_fma(partial[g],factors[g],sums[g%4])
 return _add(_add(_add(sums[0],sums[1]),_add(sums[2],sums[3])),bias)

# This full original-byte fixture realizes the old large terms with legal
# exact K128 FP8 partials ±1, rather than the old synthetic partial ±2^30.
g,n=17,129
x=torch.zeros(1,g*128,dtype=torch.bfloat16)
x.reshape(g,128)[:,0]=7;x.reshape(g,128)[:,1]=2
w=torch.zeros(n,g*128,dtype=torch.float32)
for group,sign in ((0,1),(1,-1),(2,1),(8,1)):w[:,group*128+1]=sign/32
w=w.to(torch.float8_e4m3fn);scales=torch.ones(2,g)
scales[:,[0,1]]=2.**34;scales[:,[2,8]]=16
ref=build_reference(x,w,scales)
assert ref['all_orders_exact'].all() and not ref['partial_bound'].any()
assert ref['representation_loss']['weight_changed_values']==0 and ref['representation_loss']['activation_changed_values']==0
for tensor in (ref['q_native'],ref['prepared_weight']):
 codes=tensor.numpy() & 127
 assert np.all((codes==0)|(codes>=8)), 'construct normal native FP8 operands, not FP8 subnormals'
part=ref['partial_balanced'].numpy()[:,0,:];factor=ref['factors'];bias=ref['bias'].numpy()
assert np.array_equal(part[[0,1,2,8],0],np.array([1,-1,1,1],np.float32))
terms=part.astype(np.float64)*factor.astype(np.float64)
exact=terms.sum(0);absolute=np.abs(terms).sum(0)
y=chain4(part,factor,bias);assert np.all(y==1) and np.all(exact==2)
for name,array in [('partial',part[:,None,:]),('activation-scales',ref['activation_scales'].numpy()[:,:,0]),
                   ('weight-scales',ref['prepared_scales'].numpy()),('bias',bias)]:array.astype(np.float32).tofile(args.out/(name+'.bin'))
for mode,reduction in ((1,'sequential'),(0,'neumaier_fp32')):
 output=epilogue_certificate(part,factor,bias,reduction);assert torch.all(output==2)
 output.view(torch.int16).numpy().tofile(args.out/f'cpu-chain{mode}-bf16.bin')
_bf16(y).view(torch.int16).numpy().tofile(args.out/'cpu-chain4-bf16.bin')
torch.save(dict(x=x,weight=w,scales=scales,q=ref['q_native'],activation_scales=ref['activation_scales'],
                partial=ref['partial_balanced'],factors=torch.from_numpy(factor)),args.out/'full-mac-fixture.pt')
old=args.archive/'fixtures-v2/order-sensitive-large-range'
old_output=np.fromfile(old/'cpu-chain4-f32.bin',np.float32)
assert np.all(old_output==1)
old_device=args.archive/'target-24af322/results/reduce-comparison-v1/details/order-sensitive-large-range-chain4-bf16.bin'
assert old_device.read_bytes()==(args.out/'cpu-chain4-bf16.bin').read_bytes()
report=dict(classification='DECLARED_ORDINARY_FP32_CANCELLATION_NOT_AN_ILLEGAL_PRECISION_STEP',
 old_verdict_unchanged=True,new_device_test=False,qualified_replacement=False,cpu_backend=backend,
 full_mac_fixture=dict(shape=[1,n,g*128],all_partial_dots_exact=True,all_nonzero_native_fp8_operands_normal=True,representation_loss=ref['representation_loss'],
   nonzero_partial_group0_1_2_8=part[[0,1,2,8],0].tolist(),nonzero_terms=terms[[0,1,2,8],0].tolist(),
   exact_reduction=float(exact[0]),chain4=float(y[0]),sequential=2.,neumaier=2.,absolute_error=1.,relative_error=.5,
   absolute_term_sum=float(absolute[0]),condition_number=float(absolute[0]/abs(exact[0])),
   backward_error=float(1/absolute[0]),gamma16_absolute_bound=float(_gamma(16)*absolute[0]),
   bound_scope='17 exact products; conservative 16-addition forward bound for reduction order only, not complete MAC error'),
 source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old/'partial.bin',old/'activation-scales.bin',old/'weight-scales.bin',old/'cpu-chain4-f32.bin',old_device]},
 scope='A permissible FP32 tree can have large relative error under cancellation. This proves the example arithmetic, not arbitrary-input correctness, model quality or speed.')
# Re-evaluate all columns of the already measured complete-MAC QKV fixture.
case=args.archive/'fixtures-v2/qkv-measured-m1'
p=np.fromfile(case/'partial.bin',np.float32).reshape(48,1,3392)[:,0,:]
sa=np.fromfile(case/'activation-scales.bin',np.float32).reshape(48,1)
sw=np.fromfile(case/'weight-scales.bin',np.float32).reshape(27,48)
b=np.fromfile(case/'bias.bin',np.float32)
factors=(sa*sw[np.arange(3392)//128].T).astype(np.float32)
actual=chain4(p,factors,b)
old=np.fromfile(case/'cpu-chain4-f32.bin',np.float32).reshape(1,3392)[0]
assert np.array_equal(actual.view(np.uint32),old.view(np.uint32))
full=np.fromfile(case/'reference-fp64.bin',np.float64).reshape(1,3392)[0]
report['saved_real_mme_qkv']=dict(columns=3392,independent_replay_matches_old_declared_fp32_bits=True,
 relative_l2_fp64_diagnostic=float(np.linalg.norm(actual.astype(np.float64)-full)/np.linalg.norm(full)),
 max_abs_fp64_diagnostic=float(np.max(np.abs(actual.astype(np.float64)-full))),
 scope='all saved actual MME partials; FP64 diagnostic, not new model acceptance')
(args.out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('source_files','cpu_backend')},indent=2))
