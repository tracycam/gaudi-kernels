"""Meaningful guards for attribution and route-order reporting; CPU only."""
import copy
import torch
from analyze import compare, metric

torch.set_num_threads(1)
x=torch.tensor([-1.,0.,1.],dtype=torch.bfloat16)
y=torch.nextafter(x,torch.full_like(x,float('inf')))
assert metric(x,y)['max_bf16_ulp_distance']==1
root='model.layers.0.'
stages={root+'input_layernorm':torch.ones(1,2,dtype=torch.bfloat16),
        root+'self_attn.qkv_proj':torch.ones(1,2,dtype=torch.bfloat16),
        root+'mlp.gate':torch.tensor([[1.,2.,3.]]),
        root+'mlp.experts.moe_op.route_ids':torch.tensor([[1,2]],dtype=torch.int32),
        root+'mlp.experts.moe_op.route_weights':torch.tensor([[.25,.75]])}
a=dict(stages=stages,input_ids=torch.tensor([[5]]),positions=torch.tensor([[2]]))
b=copy.deepcopy(a);b['stages'][root+'self_attn.qkv_proj'][0,0]=y[-1]
b['stages'][root+'mlp.experts.moe_op.route_ids']=torch.tensor([[2,1]],dtype=torch.int32)
b['stages'][root+'mlp.experts.moe_op.route_weights']=torch.tensor([[.75,.25]])
r=compare(a,b,1)
assert r['first_observed_difference']['observed_qkv_input_equal']
assert r['changed_route_orders']==1 and r['changed_route_sets']==0
assert r['routes'][0]['token_rows'][0]['aligned_common_weight_max_abs']==0
b['stages'][root+'input_layernorm'][0,0]=2
assert not next(v for v in compare(a,b,1)['rows'] if v['stage']=='self_attn.qkv_proj')['observed_qkv_input_equal']
for mutate in ['missing','input','dtype']:
 b=copy.deepcopy(a)
 if mutate=='missing':b['stages'].pop(root+'mlp.gate')
 if mutate=='input':b['input_ids']+=1
 if mutate=='dtype':b['stages'][root+'self_attn.qkv_proj']=b['stages'][root+'self_attn.qkv_proj'].float()
 try:compare(a,b,1)
 except ValueError:pass
 else:raise AssertionError('invalid capture accepted: '+mutate)
print('PASS: BF16 signed ULP, same-input isolation, route permutation/aligned weights, missing/input/dtype rejection')
