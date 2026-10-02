"""Opt-in pinned HPUAttentionImpl decode adapter; import never patches vLLM.

Call install_vllm_swa() once before capture. set_swa_policy() requires the caller
to invalidate all existing HPU graphs; captured recipes cannot change policy.
Unsupported calls invoke the exact forward callable present at installation.
No runtime exception is swallowed after cache writes or graph contribution.
"""
import torch
from .vllm_swa import eligible_window_call, flat_pa_window

_POLICY='vendor'
_ORIGINAL=None
_FALLBACK_REASONS={p:{} for p in ('vendor','fp32','fp32_fast','fp32_quad','fp32_av_hoist','fp32_batch','fp32_batch_quad')}
_FIRST_FALLBACK={p:{} for p in ('vendor','fp32','fp32_fast','fp32_quad','fp32_av_hoist','fp32_batch','fp32_batch_quad')}
_COUNTS={p:{'eligible_custom_calls':0,'structural_fallback_calls':0,'vendor_policy_calls':0}
         for p in ('vendor','fp32','fp32_fast','fp32_quad','fp32_av_hoist','fp32_batch','fp32_batch_quad')}


def set_swa_policy(policy):
    global _POLICY
    if policy not in ('vendor','fp32','fp32_fast','fp32_quad','fp32_av_hoist','fp32_batch','fp32_batch_quad'):
        raise ValueError('SWA policy must be vendor, fp32, fp32_fast, fp32_quad, or fp32_av_hoist')
    if policy=='fp32_quad':
        try:getattr(torch.ops.gaudi_swa128_ilp,'quad')
        except AttributeError as error:
            raise RuntimeError('Load the independent SWA quad Torch library before selecting fp32_quad') from error
    if policy in ('fp32_av_hoist','fp32_batch','fp32_batch_quad'):
        try:getattr(torch.ops.gaudi_swa128_avgroup,'group')
        except AttributeError as error:
            raise RuntimeError('Load the independent SWA AV-hoist library before selecting fp32_av_hoist') from error
    if policy in ('fp32_batch','fp32_batch_quad'):
        getattr(torch.ops.gaudi_swa128_batch,'window_quad_fp32' if policy=='fp32_batch_quad' else 'window_fp32')
    _POLICY=policy


def prepare_vllm_swa(model):
    """Inventory attention and bind existing partial-RoPE shape dispatch.

    Call before capture or after its owner clears old graphs. No weights or
    tensor contents are modified; dynamic call guards remain required.
    """
    from vllm_gaudi.attention.backends.hpu_attn import HPUAttentionImpl
    names=[]
    for name,module in model.named_modules():
        if (isinstance(module,HPUAttentionImpl) and module.sliding_window==128
            and module.num_heads==16 and module.num_kv_heads==1 and module.head_size==192
            and module.head_size_v==128 and not module.enable_fp8_attn
            and module.alibi_slopes is None and not module.is_chunked_attention
            and isinstance(module.sinks,torch.Tensor) and module.sinks.dtype==torch.bfloat16
            and not module.sinks.requires_grad):
            names.append(name)
    from .vllm_swa_rotary import prepare_rotary_boundary
    rotary=prepare_rotary_boundary(model)
    return {'matched':len(names),'modules':names,'policy':_POLICY,'rotary_boundary':rotary,
            'scope':'static module geometry only; each decode call still validates tensors and metadata'}


def snapshot_swa_counters(reset=False):
    import copy
    from .vllm_swa_rotary import rotary_boundary_snapshot
    result={'policy':_POLICY,'by_policy':{p:dict(v) for p,v in _COUNTS.items()},
            'fallback_reasons':copy.deepcopy(_FALLBACK_REASONS),'first_fallback_by_reason':copy.deepcopy(_FIRST_FALLBACK),
            'rotary_boundary':rotary_boundary_snapshot(reset=reset),
            'scope':'Python forward dispatches/capture construction only; custom count includes attempts. HPU graph replays bypass these counters; these are not device calls or executed layer counts.'}
    if reset:
        for row in _COUNTS.values():
            for key in row:row[key]=0
        for rows in (_FALLBACK_REASONS,_FIRST_FALLBACK):
            for p in rows:rows[p].clear()
    return result


def _owned_shape(x,shape):
    return x if tuple(x.shape)==tuple(shape) else torch.ops.gaudi_swa128.reshape(x,list(shape))


def _fallback_reason(self,query,key,value,kv_cache,md,output):
    from vllm.v1.attention.backend import AttentionType
    checks=[('vendor_policy',_POLICY=='vendor'),('output_buffer_present',output is not None),
            ('attention_type',self.attn_type!=AttentionType.DECODER),('sliding_window',self.sliding_window!=128),
            ('num_heads',self.num_heads!=16),('num_kv_heads',self.num_kv_heads!=1),
            ('head_size',self.head_size!=192),('value_head_size',self.head_size_v!=128),
            ('fp8_attention',self.enable_fp8_attn),('alibi',self.alibi_slopes is not None),
            ('chunked_attention',self.is_chunked_attention),('prompt',md.is_prompt)]
    for name,reject in checks:
        if reject:return name
    if not isinstance(query,torch.Tensor):return 'query_not_tensor'
    if query.ndim not in (2,3) or tuple(query.shape[1:])not in ((3072,),(1,3072)):return 'query_shape'
    rows=query.shape[0]
    if not 1<=rows<=(32 if _POLICY in ('fp32_batch','fp32_batch_quad') else 1):return 'query_shape'
    if not isinstance(kv_cache,tuple):return 'cache_container_not_tuple'
    if len(kv_cache)!=4:return 'cache_container_length'
    if kv_cache[2] is not None or kv_cache[3] is not None:return 'kv_scales_present'
    for name,t in [('query',query),('key_cache',kv_cache[0]),('value_cache',kv_cache[1]),('sinks',self.sinks)]:
        if not isinstance(t,torch.Tensor):return name+'_not_tensor'
        if t.dtype!=torch.bfloat16:return name+'_dtype'
        if t.device.type!='hpu' or t.device!=query.device:return name+'_device'
        if not t.is_contiguous():return name+'_noncontiguous'
        if t.requires_grad:return name+'_requires_grad'
    k,v=kv_cache[:2]
    if k.ndim!=3 or tuple(k.shape[1:])!=(1,192):return 'key_cache_shape'
    if v.ndim!=3 or tuple(v.shape[1:])!=(1,128):return 'value_cache_shape'
    if k.shape[0]!=v.shape[0]:return 'kv_cache_length_mismatch'
    if not 0<k.shape[0]<=2147483647//192 or k.shape[0]%128:return 'cache_capacity'
    if tuple(self.sinks.shape)!=(16,):return 'sink_shape'
    if md.block_size!=128:return 'block_size'
    blocks=getattr(md,'window_block_list',None);groups=getattr(md,'window_block_groups',None)
    pos=getattr(md,'input_positions',None);mapping=getattr(md,'window_block_mapping',None)
    for name,t in [('window_blocks',blocks),('window_groups',groups),('input_positions',pos)]:
        if not isinstance(t,torch.Tensor):return name+'_not_tensor'
        if t.dtype not in (torch.int32,torch.int64):return name+'_dtype'
        if t.device!=query.device:return name+'_device'
        if not t.is_contiguous():return name+'_noncontiguous'
    if blocks.ndim!=1 or not 0<blocks.numel()<=2147483647:return 'window_blocks_shape'
    if groups.shape!=blocks.shape:return 'window_groups_shape'
    if pos.ndim not in (1,2) or pos.numel()!=rows:return 'input_positions_shape'
    if not isinstance(mapping,torch.Tensor):return 'window_mapping_not_tensor'
    if tuple(mapping.shape)!=(blocks.numel(),rows):return 'window_mapping_shape'
    if _POLICY in ('fp32_batch','fp32_batch_quad') and blocks.numel()>4096:return 'batch_window_capacity'
    for name,t,n in [('key',key,192),('value',value,128)]:
        if t is None:continue
        if not isinstance(t,torch.Tensor):return name+'_not_tensor'
        if t.dtype!=torch.bfloat16:return name+'_dtype'
        if t.device!=query.device:return name+'_device'
        if not t.is_contiguous():return name+'_noncontiguous'
        if t.requires_grad:return name+'_requires_grad'
        if t.numel()!=rows*n:return name+'_numel'
    if self.kv_sharing_target_layer_name is None and (key is not None or value is not None):
        slot=getattr(md,'slot_mapping',None)
        if not isinstance(slot,torch.Tensor):return 'slot_not_tensor'
        if slot.numel()!=rows:return 'slot_numel'
        if slot.dtype not in (torch.int32,torch.int64):return 'slot_dtype'
        if slot.device!=query.device:return 'slot_device'
        if not slot.is_contiguous():return 'slot_noncontiguous'
    return None


def _eligible(self,query,key,value,kv_cache,md,output):
    return _fallback_reason(self,query,key,value,kv_cache,md,output) is None


def _describe_tensor(t):
    if not isinstance(t,torch.Tensor):return {'type':type(t).__name__}
    return {'type':'Tensor','shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),
            'stride':list(t.stride()),'storage_offset':t.storage_offset(),
            'contiguous':t.is_contiguous(),'requires_grad':t.requires_grad}


def _fallback_sample(self,layer,query,key,value,kv_cache,md,output):
    return {'layer_name':str(getattr(layer,'layer_name','unknown')),
            'config':{'heads':self.num_heads,'kv_heads':self.num_kv_heads,'head_size':self.head_size,
                      'value_head_size':self.head_size_v,'sliding_window':self.sliding_window,
                      'is_prompt':md.is_prompt,'block_size':getattr(md,'block_size',None)},
            'tensors':{name:_describe_tensor(t) for name,t in
                       [('query',query),('key',key),('value',value),('sinks',self.sinks),('output',output)]},
            'cache_type':type(kv_cache).__name__,
            'cache_length':len(kv_cache) if isinstance(kv_cache,(tuple,list)) else None,
            'cache_elements':[_describe_tensor(t) for t in kv_cache[:4]] if isinstance(kv_cache,(tuple,list)) else [_describe_tensor(kv_cache)],
            'metadata':{name:_describe_tensor(getattr(md,name,None)) for name in
                        ('window_block_list','window_block_groups','window_block_mapping','window_attn_bias','input_positions','slot_mapping')},
            'scope':'host-side tensor metadata only; no tensor values, device readback, item(), or data_ptr()'}


def install_vllm_swa(policy='vendor'):
    """Install a disabled-by-default decode wrapper; return integration metadata."""
    global _ORIGINAL
    from vllm_gaudi.attention.backends.hpu_attn import HPUAttentionImpl, _set_fetch_by_id
    set_swa_policy(policy)
    from .vllm_swa_rotary import install_rotary_boundary
    install_rotary_boundary(lambda:_POLICY)
    if _ORIGINAL is not None:
        return {'installed':True,'policy':_POLICY,'already_installed':True}
    original=HPUAttentionImpl.forward

    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output=None):
        if _POLICY=='vendor':
            _COUNTS[_POLICY]['vendor_policy_calls']+=1
            return original(self,layer,query,key,value,kv_cache,attn_metadata,output)
        reason=_fallback_reason(self,query,key,value,kv_cache,attn_metadata,output)
        if reason is not None:
            _COUNTS[_POLICY]['structural_fallback_calls']+=1
            reasons=_FALLBACK_REASONS[_POLICY];reasons[reason]=reasons.get(reason,0)+1
            first=_FIRST_FALLBACK[_POLICY]
            if reason not in first:first[reason]=_fallback_sample(self,layer,query,key,value,kv_cache,attn_metadata,output)
            return original(self,layer,query,key,value,kv_cache,attn_metadata,output)
        _COUNTS[_POLICY]['eligible_custom_calls']+=1
        md=attn_metadata;output_shape=(*query.shape[:-1],2048)
        rows=query.shape[0];q=_owned_shape(query,(rows,1,3072));kc,vc=kv_cache[:2]
        if self.kv_sharing_target_layer_name is None:
            slot=getattr(md,'slot_mapping',None)
            if slot is not None:slot=_owned_shape(slot,(rows,))
            kk=None if key is None else _owned_shape(key,(rows,1,192))
            vv=None if value is None else _owned_shape(value,(rows,1,128))
            # Preserve the actual vendor cache modules and their in-place
            # index_copy_ writes. Never materialize or clone a complete cache.
            kc=self.k_cache(kk,kc,slot,scales=None,block_size=128,is_prompt=False)
            vc=self.v_cache(vv,vc,slot,scales=None,block_size=128,is_prompt=False)
        _set_fetch_by_id(self.k_cache,True);_set_fetch_by_id(self.v_cache,True)
        self.position_bias=None
        kwargs=self.common_attention_args(md.window_block_list,kc,vc,128,None,None)
        kwargs.update(query=q,block_mapping=md.window_block_mapping,block_bias=md.window_attn_bias,
                      block_groups=md.window_block_groups,position_bias=None)
        assert eligible_window_call(kwargs,md.window_block_list,md.window_block_groups,md.input_positions,128,batch=_POLICY in ('fp32_batch','fp32_batch_quad'))
        context=flat_pa_window(window_block_list=md.window_block_list,window_block_groups=md.window_block_groups,
                               input_positions=md.input_positions,sliding_window=128,policy=_POLICY,**kwargs)
        return _owned_shape(context,output_shape)

    forward._gaudi_swa128_original=original
    HPUAttentionImpl.forward=forward;_ORIGINAL=original
    return {'installed':True,'policy':_POLICY,'already_installed':False,
            'scope':'M1 H16 D192 V128 BF16 SWA128 decode; graph-owned Q/K/V/output shape edges; original cache modules',
            'graph_invalidation_required_on_policy_change':True}
