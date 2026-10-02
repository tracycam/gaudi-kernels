"""Opt-in pinned MiMo TP8 decode postprocessing; import changes nothing.

The caller loads the libraries and invalidates captured graphs when changing
policy. The wrapper keeps both projections and the existing Attention module,
including its cache writes and SWA dispatch. All fallbacks precede qkv_proj.
"""
import functools
import inspect
import sys
import warnings
import torch
from .qkv_postprocess import qkv_postprocess_cache

_POLICY='vendor'
_ATTENTION_SCOPE='swa128'
_MAX_ROWS=1
_PREPARED_ATTENTION_COUNTS=None
_TARGET=_ROPE=_ATTN=_IMPL=_DIFF_IMPL=_QKV=_ROW=None
_ORIGINAL=_ROTARY_BASE=None
_COUNTS={p:{'vendor_policy_calls':0,'eligible_custom_calls':0,'structural_fallback_calls':0,'eligible_swa128_calls':0,'eligible_full_calls':0}
         for p in ('vendor','fused')}
_REASONS={p:{} for p in _COUNTS}
_FIRST={p:{} for p in _COUNTS}
_ELIGIBLE_LAYERS={p:{'swa128':{},'full':{}} for p in _COUNTS}


def _rotary_hook_contract(rope):
    """Accept only no hooks or the installed SDK's exact, complete naming pair."""
    state=torch.nn.modules.module
    if getattr(state,'_global_forward_hooks',{}) or getattr(state,'_global_forward_pre_hooks',{}):
        return 'unsupported_global_hooks',None
    pre=getattr(rope,'_forward_pre_hooks',{})
    post=getattr(rope,'_forward_hooks',{})
    if not pre and not post:return 'none',None
    sdk=sys.modules.get('habana_frameworks.torch.core.torch_overwrites')
    before=getattr(sdk,'_pre_fwd_hook',None);after=getattr(sdk,'_post_fwd_hook',None)
    if (len(pre)!=1 or len(post)!=1 or not callable(before) or not callable(after)
            or next(iter(pre.values())) is not before or next(iter(post.values())) is not after):
        return 'unsupported_module_hooks',None
    post_id=next(iter(post))
    always=getattr(rope,'_forward_hooks_always_called',{})
    if (getattr(rope,'_forward_pre_hooks_with_kwargs',{})
            or getattr(rope,'_forward_hooks_with_kwargs',{})
            or set(always)!={post_id} or always[post_id] is not True):
        return 'unsupported_hook_flags',None
    if getattr(rope,'names_hook',False) is not True or not isinstance(getattr(rope,'custom_name',None),str):
        return 'unsupported_sdk_marker',None
    return 'sdk_naming_pair',(before,after)


def _hook_identity(hook):
    target=getattr(hook,'__func__',hook)
    try:sourcefile=inspect.getsourcefile(target)
    except (TypeError,OSError):sourcefile=None
    code=getattr(target,'__code__',None)
    return {'module':getattr(target,'__module__',type(target).__module__),
            'qualname':getattr(target,'__qualname__',type(target).__qualname__),
            'sourcefile':sourcefile,'firstlineno':getattr(code,'co_firstlineno',None),
            'callable_type':type(hook).__qualname__}


def _hook_records(hooks,with_kwargs=(),always_called=()):
    return [{'id':key,**_hook_identity(hook),'with_kwargs':key in with_kwargs,
             'always_called':key in always_called} for key,hook in tuple(hooks.items())]


def _rotary_hook_inventory(rope):
    custom_name=getattr(rope,'custom_name',None)
    return {'contract':_rotary_hook_contract(rope)[0],
            'module_type':type(rope).__module__+'.'+type(rope).__qualname__,
            'custom_name':custom_name if isinstance(custom_name,str) else None,
            'custom_name_type':type(custom_name).__qualname__,'names_hook':getattr(rope,'names_hook',False) is True,
            'pre':_hook_records(getattr(rope,'_forward_pre_hooks',{}),getattr(rope,'_forward_pre_hooks_with_kwargs',{})),
            'post':_hook_records(getattr(rope,'_forward_hooks',{}),getattr(rope,'_forward_hooks_with_kwargs',{}),
                                 getattr(rope,'_forward_hooks_always_called',{}))}


def _postprocess_with_sdk_hooks(rope,qkv,positions,scale,pair):
    if pair is None:return qkv_postprocess_cache(qkv,rope.cos_sin_cache,positions,scale)
    before,after=pair
    # The pinned SDK naming pair never inspects its input tuple. It does inspect
    # output requires_grad, which this inference path preserves as false. No
    # packed-QKV views are materialized just to construct unused hook arguments.
    hook_input=(positions,qkv,None);hook_output=None;failed=False
    try:
        before(rope,hook_input)
        q,k,v=qkv_postprocess_cache(qkv,rope.cos_sin_cache,positions,scale)
        hook_output=(q,k)
        return q,k,v
    except BaseException:
        failed=True
        raise
    finally:
        if not failed:after(rope,hook_input,hook_output)
        else:
            # Match Module's always_call behavior: a secondary post-hook error
            # must not replace the original forward exception.
            try:after(rope,hook_input,hook_output)
            except Exception as error:
                warnings.warn('SDK rotary post-hook failed while preserving the original exception: '+str(error),RuntimeWarning)


def set_qkv_postprocess_policy(policy,*,attention_scope='swa128',max_rows=1):
    """Caller must synchronize and invalidate/rebuild existing captured graphs."""
    global _POLICY,_ATTENTION_SCOPE,_PREPARED_ATTENTION_COUNTS,_MAX_ROWS
    if type(max_rows) is not int or max_rows not in (1,16):raise ValueError('unknown QKV postprocess row capacity')
    if attention_scope not in ('swa128','swa128_or_full'):raise ValueError('unknown QKV attention scope')
    if policy not in _COUNTS:raise ValueError('QKV postprocess policy must be vendor or fused')
    if policy=='fused' and not hasattr(torch.ops.gaudi_kernels,'_qkv_post_cache_bf16_v2'):
        raise RuntimeError('load the QKV postprocess Torch/TPC libraries before enabling fused policy')
    previous=_POLICY;previous_scope=_ATTENTION_SCOPE;previous_rows=_MAX_ROWS
    _POLICY=policy;_ATTENTION_SCOPE=attention_scope;_MAX_ROWS=max_rows
    _PREPARED_ATTENTION_COUNTS=None
    return {'previous':previous,'policy':policy,'previous_attention_scope':previous_scope,'attention_scope':attention_scope,
            'previous_max_rows':previous_rows,'max_rows':max_rows,
            'graph_invalidation_required':previous!=policy or previous_scope!=attention_scope or previous_rows!=max_rows}


def _known_rotary_dispatch(rope):
    bound=getattr(rope,'_forward_method',None)
    if getattr(bound,'__self__',None) is not rope:return False
    fn=getattr(bound,'__func__',None)
    if fn is _ROTARY_BASE:return True
    # This known adapter changes the final shape edge while retaining the
    # actual vendor arithmetic. Accept only its exact callable, not arbitrary
    # __wrapped__ chains or functions merely named forward_oot.
    swa=sys.modules.get('gaudi_kernels.vllm_swa_rotary')
    return (swa is not None and getattr(swa,'_ORIGINAL',None) is _ROTARY_BASE
            and fn is getattr(swa,'_WRAPPER',None))


def _static_reason(module):
    if type(module) is not _TARGET:return 'attention_module_class'
    if 'forward' in module.__dict__:return 'instance_forward_override'
    for name,value in [('hidden_size',6144),('num_heads',16),('num_kv_heads',1),
                       ('total_num_heads',128),('total_num_kv_heads',8),
                       ('head_dim',192),('v_head_dim',128),
                       ('q_size',3072),('k_size',192),('v_size',128)]:
        if getattr(module,name,None)!=value:return 'geometry_'+name
    if type(module.v_scale) is not float or module.v_scale!=.612:return 'value_scale'
    qkv=getattr(module,'qkv_proj',None);row=getattr(module,'o_proj',None)
    if type(qkv) is not _QKV or type(row) is not _ROW:return 'projection_class'
    for name,value in [('input_size',6144),('output_size_per_partition',3392),
                       ('tp_size',8),('gather_output',False),('return_bias',True)]:
        if getattr(qkv,name,None)!=value:return 'qkv_'+name
    if tuple(getattr(qkv,'output_partition_sizes',()))!=(3072,192,128):return 'qkv_partitions'
    if (getattr(row,'input_size_per_partition',None)!=2048
            or getattr(row,'output_size_per_partition',None)!=6144
            or getattr(row,'tp_size',None)!=8 or not getattr(row,'return_bias',False)):
        return 'output_projection_geometry'
    rope=getattr(module,'rotary_emb',None)
    if type(rope) is not _ROPE:return 'rotary_class'
    # Unknown hooks can replace inputs/outputs. The one qualified SDK pair is
    # explicitly executed around the fused call; it is never simply ignored.
    if _rotary_hook_contract(rope)[0].startswith('unsupported'):
        return 'rotary_forward_hooks'
    if (rope.head_size,rope.rotary_dim,rope.is_neox_style)!=(192,64,True):return 'rotary_geometry'
    if not _known_rotary_dispatch(rope):return 'rotary_dispatch'
    if hasattr(rope,'scaling_factor') or hasattr(rope,'scaling_factors'):return 'rotary_scaling_override'
    cache=getattr(rope,'cos_sin_cache',None)
    if not isinstance(cache,torch.Tensor):return 'rotary_cache_not_tensor'
    if cache.dtype!=torch.bfloat16:return 'rotary_cache_dtype'
    if cache.ndim!=2 or cache.shape[1]!=64 or not 0<cache.shape[0]<=2147483647//64:return 'rotary_cache_shape'
    if not cache.is_contiguous() or cache.requires_grad:return 'rotary_cache_layout_or_grad'
    attn=getattr(module,'attn',None)
    if type(attn) is not _ATTN:return 'plugin_attention_class'
    if not getattr(attn,'use_direct_call',False):return 'attention_indirect_call'
    if getattr(attn,'query_quant',None) is not None:return 'query_quantization'
    impl=getattr(attn,'impl',None)
    if type(impl) not in (_IMPL,_DIFF_IMPL):return 'attention_impl_class'
    if type(impl) is _DIFF_IMPL and _DIFF_IMPL.forward is not _IMPL.forward:return 'diffkv_forward_override'
    window=getattr(impl,'sliding_window','missing')
    if window!=128 and not (_ATTENTION_SCOPE=='swa128_or_full' and window is None):return 'attention_sliding_window'
    for name,value in [('num_heads',16),('num_kv_heads',1),
                       ('head_size',192),('head_size_v',128),('enable_fp8_attn',False),
                       ('is_chunked_attention',False),('alibi_slopes',None),
                       ('kv_sharing_target_layer_name',None),('attn_type','decoder')]:
        if getattr(impl,name,None)!=value:return 'attention_'+name
    return None


def _fallback_reason(module,positions,hidden_states):
    reason=_static_reason(module)
    if reason is not None:return reason
    if not isinstance(hidden_states,torch.Tensor):return 'hidden_not_tensor'
    if hidden_states.ndim!=2 or hidden_states.shape[1]!=6144 or not 1<=hidden_states.shape[0]<=_MAX_ROWS:return 'hidden_shape'
    rows=hidden_states.shape[0]
    if hidden_states.dtype!=torch.bfloat16:return 'hidden_dtype'
    if hidden_states.device.type!='hpu':return 'hidden_device'
    if not hidden_states.is_contiguous() or hidden_states.requires_grad:return 'hidden_layout_or_grad'
    cache=module.rotary_emb.cos_sin_cache
    if cache.device!=hidden_states.device:return 'rotary_cache_device'
    if not isinstance(positions,torch.Tensor):return 'positions_not_tensor'
    if tuple(positions.shape) not in ((rows,),(rows,1),(1,rows)):return 'positions_shape'
    if positions.dtype not in (torch.int32,torch.int64):return 'positions_dtype'
    if positions.device!=hidden_states.device:return 'positions_device'
    if not positions.is_contiguous():return 'positions_layout'
    from vllm.forward_context import get_forward_context,is_forward_context_available
    if not is_forward_context_available():return 'forward_context_absent'
    metadata=get_forward_context().attn_metadata
    if isinstance(metadata,dict):metadata=metadata.get(module.attn.layer_name)
    if metadata is None:return 'layer_metadata_absent'
    if getattr(metadata,'is_prompt',True):return 'prompt'
    if getattr(metadata,'block_size',None)!=128:return 'metadata_block_size'
    return None


def _describe(tensor):
    if not isinstance(tensor,torch.Tensor):return {'type':type(tensor).__name__}
    return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'device':str(tensor.device),
            'stride':list(tensor.stride()),'contiguous':tensor.is_contiguous(),
            'requires_grad':tensor.requires_grad}


def _record_eligible(module):
    kind='swa128' if module.attn.impl.sliding_window==128 else 'full'
    _COUNTS['fused']['eligible_custom_calls']+=1
    _COUNTS['fused']['eligible_'+kind+'_calls']+=1
    name=str(module.attn.layer_name);row=_ELIGIBLE_LAYERS['fused'][kind];row[name]=row.get(name,0)+1


def attention_from_qkv(module,positions,qkv,*,count_external=False):
    """Consume an explicitly preprojected QKV after pre-producer qualification.

    No projection or fallback occurs here. The caller must have checked the
    existing module/input/metadata guards before its norm/projection work.
    """
    if _POLICY!='fused':raise RuntimeError('explicit QKV consumer requires fused postprocess policy')
    if (qkv.ndim!=2 or qkv.shape[1]!=3392 or not 1<=qkv.shape[0]<=_MAX_ROWS or qkv.shape[0]!=positions.numel() or qkv.dtype!=torch.bfloat16
            or qkv.device!=positions.device or not qkv.is_contiguous() or qkv.requires_grad):
        raise RuntimeError('qualified QKV projection violated its bounded contiguous BF16 [M,3392] output contract')
    from .serving.diagnostics.boundary_hashes import emit
    hook_mode,hook_pair=_rotary_hook_contract(module.rotary_emb)
    if hook_mode.startswith('unsupported'):raise RuntimeError('rotary hooks changed after pre-producer qualification')
    if count_external:_record_eligible(module)
    q,k,v=_postprocess_with_sdk_hooks(module.rotary_emb,qkv,positions,module.v_scale,hook_pair)
    context=module.attn(q,k,v)
    output,_=module.o_proj(context)
    emit(module.attn.layer_name,'attention',output)
    return output


def snapshot_qkv_postprocess(reset=False):
    import copy
    result={'policy':_POLICY,'attention_scope':_ATTENTION_SCOPE,'max_rows':_MAX_ROWS,'prepared_attention_counts':copy.deepcopy(_PREPARED_ATTENTION_COUNTS),
            'eligible_layers_by_policy':copy.deepcopy(_ELIGIBLE_LAYERS),'by_policy':copy.deepcopy(_COUNTS),
            'fallback_reasons':copy.deepcopy(_REASONS),'first_fallback_by_reason':copy.deepcopy(_FIRST),
            'scope':'Python forward/capture dispatches; eligible count includes attempts. Graph replays bypass these counters, so they are not device or layer execution counts.'}
    if reset:
        for row in _COUNTS.values():
            for key in row:row[key]=0
        for kinds in _ELIGIBLE_LAYERS.values():
            for row in kinds.values():row.clear()
        for rows in (_REASONS,_FIRST):
            for row in rows.values():row.clear()
    return result


def prepare_vllm_qkv_postprocess(model):
    """Inventory current module/cache/dispatch bindings; does not mutate tensors."""
    if _TARGET is None:raise RuntimeError('install the QKV postprocess wrapper before preparing a model')
    global _PREPARED_ATTENTION_COUNTS
    matched=[];rejected={};rotary_hooks={};kinds={'swa128':[],'full':[]}
    for name,module in model.named_modules():
        if type(module) is not _TARGET:continue
        rotary_hooks[name]=_rotary_hook_inventory(getattr(module,'rotary_emb',None))
        reason=_static_reason(module)
        if reason is None:
            matched.append(name);kinds['swa128' if module.attn.impl.sliding_window==128 else 'full'].append(name)
        else:rejected[name]=reason
    hook_state=torch.nn.modules.module
    global_hooks={'pre':_hook_records(getattr(hook_state,'_global_forward_pre_hooks',{})),
                  'post':_hook_records(getattr(hook_state,'_global_forward_hooks',{}),
                                       always_called=getattr(hook_state,'_global_forward_hooks_always_called',{}))}
    _PREPARED_ATTENTION_COUNTS={kind:len(names) for kind,names in kinds.items()}
    return {'matched':len(matched),'modules':matched,'rejected':rejected,'policy':_POLICY,
            'attention_scope':_ATTENTION_SCOPE,'matched_by_attention_kind':dict(_PREPARED_ATTENTION_COUNTS),'modules_by_attention_kind':kinds,
            'rotary_hooks':rotary_hooks,'global_hooks':global_hooks,
            'scope':'static TP8 module within explicit attention scope and actual rotary dispatch/cache geometry; input and metadata guards remain dynamic; no parameters or bound methods changed'}


def install_vllm_qkv_postprocess(policy='vendor',*,model=None,attention_scope='swa128',max_rows=1):
    """Patch the exact MiMoV2Attention class; unsupported calls use its original forward."""
    global _TARGET,_ROPE,_ATTN,_IMPL,_DIFF_IMPL,_QKV,_ROW,_ORIGINAL,_ROTARY_BASE
    from vllm.model_executor.models.mimo_v2 import MiMoV2Attention
    from vllm_gaudi.ops.hpu_rotary_embedding import HPURotaryEmbedding
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm_gaudi.attention.backends.hpu_attn import HPUAttentionImpl
    from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionDiffKVImpl
    from vllm.model_executor.layers.linear import QKVParallelLinear,RowParallelLinear
    change=set_qkv_postprocess_policy(policy,attention_scope=attention_scope,max_rows=max_rows)
    if _TARGET is None:
        _TARGET,_ROPE,_ATTN,_IMPL,_QKV,_ROW=(MiMoV2Attention,HPURotaryEmbedding,Attention,
                                           HPUAttentionImpl,QKVParallelLinear,RowParallelLinear)
        _DIFF_IMPL=HPUAttentionDiffKVImpl
        _ORIGINAL=MiMoV2Attention.forward
        # Installing before or after the known graph-owned rotary adapter is
        # supported; existing rope instances may retain the original binding.
        swa=sys.modules.get('gaudi_kernels.vllm_swa_rotary')
        _ROTARY_BASE=(getattr(swa,'_ORIGINAL',None) if swa is not None else None) or HPURotaryEmbedding.forward_oot

        @functools.wraps(_ORIGINAL)
        def forward(self,positions,hidden_states):
            def original():
                output = _ORIGINAL(self,positions,hidden_states)
                from .serving.diagnostics import boundary_hashes as boundary
                if boundary.active():
                    boundary.emit(self.attn.layer_name, 'attention', output)
                return output
            if _POLICY=='vendor':
                _COUNTS[_POLICY]['vendor_policy_calls']+=1
                return original()
            reason=_fallback_reason(self,positions,hidden_states)
            if reason is not None:
                _COUNTS[_POLICY]['structural_fallback_calls']+=1
                row=_REASONS[_POLICY];row[reason]=row.get(reason,0)+1
                if reason not in _FIRST[_POLICY]:
                    _FIRST[_POLICY][reason]={'hidden':_describe(hidden_states),'positions':_describe(positions),
                                            'layer_name':str(getattr(getattr(self,'attn',None),'layer_name','unknown'))}
                return original()
            _record_eligible(self)
            hook_mode,hook_pair=_rotary_hook_contract(self.rotary_emb)
            if hook_mode.startswith('unsupported'):
                raise RuntimeError('rotary hook state changed during pre-projection qualification')
            qkv,_=self.qkv_proj(hidden_states)
            # A projection violating its declared output contract is an error,
            # never a halfway fallback or second projection/cache mutation.
            # The single consumer validates rows/dtype/storage before any cache
            # work, including calls from the fused norm producer.
            return attention_from_qkv(self,positions,qkv)

        MiMoV2Attention.forward=forward
    elif _TARGET is not MiMoV2Attention:
        raise RuntimeError('MiMoV2Attention class changed after wrapper installation')
    return {**change,'installed':True,'prepared':prepare_vllm_qkv_postprocess(model) if model is not None else None,
            'scope':'bounded BF16 TP8 MiMo rows in explicit attention_scope; original qkv_proj, attention/cache writes, and o_proj; model qualification is separate'}
