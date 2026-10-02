"""Audit-only old-norm witness and actual grid24 consumer evidence.

The serving recipe has already completed before any diagnostic outputs are
retained/read back. This helper is not called during ordinary capture/replay.
"""
import time
import torch
from .block_fp8 import linear_block_fp8_quantized
from .residual_rmsnorm import residual_rmsnorm_bf16


def active(state):
    return bool(state.audit_enabled and state.audit_frame is not None)


def record(state,layer,x,residual,gamma,epsilon,plain_r,plain_y,reducer):
    if not active(state):return
    if state.audit_contract!='fp32_arithmetic_v1':
        raise RuntimeError('grid24 requires explicit FP32-v1 audit; legacy audit must use separate policy')
    if not hasattr(layer,'_gk_oracle_weight'):
        raise RuntimeError('grid24 staged audit requires original host checkpoint weights')
    from .block_fp8_fp32_contract import build_reference,audit,recipe_relation,UnsupportedArithmetic,_sha,_reject_source_subnormal
    from .fp32_artifact_binding import verify_artifacts
    from .norm_grid24_binding import verify
    begin=time.monotonic()
    # Caller released q/sa before this point. Only real serving outputs remain.
    y_cpu=plain_y.detach().cpu().contiguous();r_cpu=plain_r.detach().cpu().contiguous()
    witness_r,yy=residual_rmsnorm_bf16(x,residual,gamma,epsilon)
    old_q,old_sa=torch.ops.gaudi_block_fp8.quant(yy)
    staged_r,q,sa=torch.ops.gaudi_norm_grid24.block128(x,residual,gamma,epsilon)
    tensors={}
    staged_y=linear_block_fp8_quantized(q,sa,layer._gk_block_fp8,
        bias=layer.bias,reduce_op=reducer,audit_tensors=tensors)
    actual={name:value.detach().cpu().contiguous()for name,value in tensors.items()}
    yy_cpu=yy.detach().cpu().contiguous()
    def equal(a,b):return recipe_relation(a,b)
    relation=equal(y_cpu,staged_y.detach().cpu().contiguous())
    producer={
        'plain_vs_staged_residual':equal(r_cpu,staged_r.detach().cpu().contiguous()),
        'old_vs_grid_residual':equal(witness_r.detach().cpu().contiguous(),r_cpu),
        'old_vs_grid_q':equal(old_q.detach().cpu().contiguous(),actual['q_native']),
        'old_vs_grid_scale':equal(old_sa.detach().cpu().contiguous(),actual['activation_scales'])}
    source={name:t.detach().cpu().contiguous()for name,t in [('x',x),('residual',residual),('gamma',gamma)]}
    try:
        for name,t in source.items():_reject_source_subnormal(t,'norm '+name)
        _reject_source_subnormal(yy_cpu,'norm BF16 witness')
        # Range only: FP64 is not the precision acceptance reference. Reject
        # square underflow/overflow before certifying the declared FP32 norm.
        rr=witness_r.detach().cpu().double();square=rr.square()
        tiny=torch.finfo(torch.float32).tiny;maximum=torch.finfo(torch.float32).max
        if (not torch.isfinite(square).all() or bool(((square!=0)&(square<tiny)).any())
                or bool((square.sum(-1)>maximum).any())):
            raise UnsupportedArithmetic('norm FP32 square/statistics range unsupported')
        producer['implementation']=verify()
        producer['passed']=all(v['all_bits_equal']for v in producer.values()if isinstance(v,dict)and'all_bits_equal'in v)
        expected=build_reference(yy_cpu,layer._gk_oracle_weight,layer._gk_oracle_scales,
            bias=None if layer.bias is None else layer.bias.detach().cpu(),native_half=True)
        checked=audit(expected,actual,reduction=state.reduction_policy,implementation=verify_artifacts(state.reduction_policy))
        if not producer['passed']:
            checked.update(passed=False,classification='GRID24_PRODUCER_RELATION_FAILED')
        elif not relation['all_bits_equal']:
            checked.update(passed=False,classification='AUDIT_RECIPE_VARIATION_INCOMPLETE')
    except UnsupportedArithmetic as error:
        checked={'contract':'block_fp8_fp32_v1','passed':False,'classification':'UNSUPPORTED_ARITHMETIC_RANGE','reason':str(error)}
    state.audit_records.append({'layer':layer.prefix,'policy':state.policy,'same_input':True,
        'quality_contract':state.audit_contract,'tag':state.audit_tag,'frame':dict(state.audit_frame),
        'input_sha256':_sha(yy_cpu),'reduction_policy':state.reduction_policy,'plain_vs_staged':relation,
        'source_layout':getattr(layer,'_gk_oracle_layout',None),
        'rounded_half_values':layer._gk_block_fp8.rounded_half_values,
        'checkpoint_weight_sha256':layer._gk_block_fp8.checkpoint_weight_sha256,
        'checkpoint_scales_sha256':layer._gk_block_fp8.checkpoint_scales_sha256,
        'fp32_contract':checked,'norm_grid24_producer':producer,
        'norm_inputs_sha256':{name:_sha(t)for name,t in source.items()},'norm_epsilon':float(epsilon),
        'diagnostic_wall_s':time.monotonic()-begin,
        'scope':'plain grid output returned; audit-only qualified old-norm BF16 witness and actual grid q/sa/P/Y; excluded from TPS'})
