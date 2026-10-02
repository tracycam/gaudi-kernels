"""Default-off exact MiMo decoder input-norm/QKV producer-consumer adapter.

No tensor identity cache. All fallbacks precede producer work. Other norms,
first-layer, M>1, A16, CPU references, and unmatched attention use the original
decoder. Only exact SDK naming hooks can cross the bypassed Module boundaries.
"""
from collections import Counter
from contextlib import ExitStack,contextmanager
import functools
import warnings
import torch
from . import vllm_norm as norm
from . import vllm_qkv_postprocess as post
from . import production_integration as block
from .block_fp8 import linear_block_fp8_quantized
from .norm_qkv_explicit import input_reason
from . import norm_grid24_audit as audit
from .serving.diagnostics import boundary_hashes as boundary

_TARGET=_ORIGINAL=None
_POLICY='separate'
_COUNTS=Counter()
_REASONS=Counter()
_FIRST={}
_BINDING=None
_AUDIT_LAYERS=[]
_QKV_FORWARD=_ATTENTION_FORWARD=None


@contextmanager
def _sdk_scope(module,args):
    mode,pair=post._rotary_hook_contract(module)
    if mode.startswith('unsupported'):raise RuntimeError('hooks changed after pre-producer qualification')
    output=[None];failed=False
    try:
        if pair:pair[0](module,args)
        yield output
    except BaseException:
        failed=True
        raise
    finally:
        if pair:
            if not failed:pair[1](module,args,output[0])
            else:
                try:pair[1](module,args,output[0])
                except Exception as error:warnings.warn('SDK post-hook failed while preserving original exception: '+str(error),RuntimeWarning)


def _sdk_call(module,args,fn):
    with _sdk_scope(module,args) as result:
        result[0]=fn()
        return result[0]


def _static(module):
    if type(module) is not _TARGET:return 'decoder_class'
    if 'forward' in module.__dict__:return 'decoder_forward_override'
    n=getattr(module,'input_layernorm',None)
    if type(n) is not norm._target:return 'input_norm_class'
    if (getattr(getattr(n,'_forward_method',None),'__func__',None) is not norm._forward
            or 'forward' in n.__dict__ or 'forward_oot' in n.__dict__):return 'input_norm_dispatch'
    if n.weight.requires_grad:return 'gamma_not_frozen'
    attention=getattr(module,'self_attn',None)
    # Grid24 remains independently qualified for SWA128 even if the shared
    # postprocess installer explicitly admits full-attention modules.
    if getattr(getattr(getattr(attention,'attn',None),'impl',None),'sliding_window',None)!=128:
        return 'post_attention_sliding_window'
    reason=post._static_reason(attention)
    if reason:return 'post_'+reason
    qkv=attention.qkv_proj
    if ('forward' in qkv.__dict__ or type(qkv).forward is not _QKV_FORWARD
            or type(attention).forward is not _ATTENTION_FORWARD):return 'projection_attention_forward_override'
    state=getattr(qkv,'_gk_block_fp8_policy_owner',None)
    if state is None or state is not block._active:return 'block_policy_owner'
    if type(qkv.quant_method) is not state.method_class or 'apply' in qkv.quant_method.__dict__:return 'block_method_override'
    if not getattr(qkv,'_gk_block_fp8_selected',False):return 'qkv_not_selected'
    if qkv.bias is not None:return 'qkv_bias_not_qualified'
    if (qkv._gk_block_fp8.n,qkv._gk_block_fp8.k)!=(3392,6144):return 'prepared_geometry'
    for name,m in [('input_norm',n),('self_attention',attention),('qkv',qkv)]:
        if post._rotary_hook_contract(m)[0].startswith('unsupported'):return name+'_hooks'
    return None


def _reason(module,positions,x,residual):
    reason=_static(module)
    if reason:return reason
    if post._POLICY!='fused':return 'qkv_post_policy'
    state=block._active
    # The small-batch policy has exactly the same M1 block-A8 contract. Keep
    # its qualified M1 fusion when a mixed request batch temporarily has one
    # active row. The producer's shape guard below still rejects M>1.
    if state.policy not in ('decode_a8_bf16_fp32','small_batch_a8_bf16_fp32'):return 'block_policy'
    n=module.input_layernorm
    reason=input_reason(x,residual,n.weight,n.variance_epsilon,norm_policy=norm.get_norm_policy(),activation_policy='per_block_fp8')
    if reason:return reason
    if getattr(n,'variance_size_override',None) is not None:return 'variance_override'
    if state.audit_enabled and state.audit_contract!='fp32_arithmetic_v1':return 'legacy_audit'
    return post._fallback_reason(module.self_attn,positions,x)


def _forward(self,positions,hidden_states,residual):
    if _POLICY=='separate':
        _COUNTS['separate_policy_calls']+=1
        if boundary.active():
            result=_ORIGINAL(self,positions,hidden_states,residual)
            boundary.emit(self.self_attn.attn.layer_name,'layer_output',result[0])
            return result
        return _ORIGINAL(self,positions,hidden_states,residual)
    reason=_reason(self,positions,hidden_states,residual)
    if reason:
        _COUNTS['fallback_calls']+=1;_REASONS[reason]+=1
        if reason not in _FIRST:_FIRST[reason]={'x':post._describe(hidden_states),'residual':post._describe(residual)}
        if boundary.active():
            result=_ORIGINAL(self,positions,hidden_states,residual)
            boundary.emit(self.self_attn.attn.layer_name,'layer_output',result[0])
            return result
        return _ORIGINAL(self,positions,hidden_states,residual)
    _COUNTS['eligible_custom_calls']+=1
    n=self.input_layernorm;att=self.self_attn;qkv=att.qkv_proj;state=block._active
    state.python_apply_branch_counts['external_norm_grid24/fp32']+=1
    reducer=state.reducer()
    with ExitStack() as stack:
        def producer_consumer():
            rr,q,sa=_sdk_call(n,(hidden_states,residual),lambda:torch.ops.gaudi_norm_grid24.block128(hidden_states,residual,n.weight,n.variance_epsilon))
            # Keep the original naming nesting: norm ends before self_attn,
            # then qkv, rotary, attention/cache writer, and output projection.
            holder=stack.enter_context(_sdk_scope(att,(positions,hidden_states)))
            pair=_sdk_call(qkv,(None,),lambda:(linear_block_fp8_quantized(q,sa,qkv._gk_block_fp8,reduce_op=reducer),None))
            return rr,pair[0],holder
        # q/sa references die with this function before diagnostic readbacks.
        rr,y,holder=producer_consumer()
        # This explicit projection bypasses Linear.apply; it owns its boundary.
        boundary.emit(qkv.prefix,'qkv',y)
        if audit.active(state):
            before=len(state.audit_records)
            audit.record(state,qkv,hidden_states,residual,n.weight,n.variance_epsilon,rr,y,reducer)
            if len(state.audit_records)!=before+1:raise RuntimeError('grid24 audit did not emit its actual producer/consumer record')
            _COUNTS['actual_grid_audit_calls']+=1
        output=post.attention_from_qkv(att,positions,y,count_external=True)
        holder[0]=output
    # Postnorm, MLP, and their Module hooks execute through the original calls.
    output,rr=self.post_attention_layernorm(output,rr)
    if boundary.active():
        result=self.mlp(output)
        boundary.emit(self.self_attn.attn.layer_name,'layer_output',result)
        return result,rr
    return self.mlp(output),rr


def snapshot(reset=False):
    result={'installed':_TARGET is not None,'policy':_POLICY,'counts':dict(_COUNTS),'expected_audit_layers':list(_AUDIT_LAYERS),
        'fallback_reasons':dict(_REASONS),'first_fallback':dict(_FIRST),'producer_binding':_BINDING,
        'scope':'Python forward/capture/audit counts, never device replay counts'}
    if reset:_COUNTS.clear();_REASONS.clear();_FIRST.clear()
    return result


def prepare(model):
    global _AUDIT_LAYERS
    if _TARGET is None:raise RuntimeError('install grid24 adapter first')
    matched=[];rejected={};audit_layers=[]
    for name,module in model.named_modules():
        if type(module) is not _TARGET:continue
        reason=_static(module)
        if reason:rejected[name]=reason
        else:
            matched.append(name)
            if getattr(module,'layer_id',0)>0:audit_layers.append(module.self_attn.qkv_proj.prefix)
    _AUDIT_LAYERS=audit_layers
    return {'matched':len(matched),'modules':matched,'rejected':rejected,'expected_audit_layers':list(audit_layers),'scope':'static SWA decoder inventory; first model layer excluded from residual-present audit coverage; input/policy guards remain dynamic'}


def install(policy='separate',*,model=None):
    global _TARGET,_ORIGINAL,_POLICY,_BINDING,_QKV_FORWARD,_ATTENTION_FORWARD
    if policy not in ('separate','grid24'):raise ValueError('unknown norm-QKV policy')
    from vllm.model_executor.models.mimo_v2 import MiMoV2FlashDecoderLayer
    if _TARGET is None:
        if post._TARGET is None:raise RuntimeError('install qualified QKV postprocess adapter before grid24')
        _TARGET=MiMoV2FlashDecoderLayer;_ORIGINAL=_TARGET.forward
        _QKV_FORWARD=post._QKV.forward;_ATTENTION_FORWARD=post._TARGET.forward
        _TARGET.forward=functools.wraps(_ORIGINAL)(_forward)
    elif _TARGET is not MiMoV2FlashDecoderLayer:raise RuntimeError('decoder class changed after grid24 installation')
    if policy=='grid24':
        from .norm_grid24_binding import verify
        _BINDING=verify()
        getattr(torch.ops.gaudi_norm_grid24,'block128')
    previous=_POLICY;_POLICY=policy
    return {'previous':previous,'policy':policy,'graph_invalidation_required':previous!=policy,
            'prepared':prepare(model)if model is not None else None,'binding':_BINDING}
