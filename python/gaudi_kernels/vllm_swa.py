"""Explicit flat_pa SWA128 adapter; import never patches or enables a policy.

The caller must pass the window metadata selected by HPUAttentionImpl and own
capture invalidation on policy changes. Full attention, prefill and unsupported
geometry keep the actual vendor flat_pa path. No metadata tensor is constructed,
cast, cloned, sliced, reshaped or copied by this entry.
"""
import torch


def eligible_window_call(kwargs, window_block_list, window_block_groups, input_positions, sliding_window, *, batch=False):
    q=kwargs.get('query');k=kwargs.get('key_cache');v=kwargs.get('value_cache');sink=kwargs.get('sinks')
    if sliding_window!=128 or kwargs.get('block_size')!=128:
        return False
    if kwargs.get('position_bias') is not None or kwargs.get('k_scales') is not None or kwargs.get('v_scales') is not None:
        return False
    if kwargs.get('block_list') is not window_block_list or kwargs.get('block_groups') is not window_block_groups:
        return False
    if any(not isinstance(t,torch.Tensor) for t in [q,k,v,sink,window_block_list,window_block_groups,input_positions]):
        return False
    if (q.ndim!=3 or tuple(q.shape[1:])!=(1,3072) or not 1<=q.shape[0]<=(32 if batch else 1)
            or tuple(sink.shape)!=(16,)):
        return False
    if k.ndim!=3 or v.ndim!=3 or tuple(k.shape[1:])!=(1,192) or tuple(v.shape[1:])!=(1,128) or k.shape[0]!=v.shape[0] or not (0<k.shape[0]<=2147483647//192) or k.shape[0]%128:
        return False
    for tensor in [q,k,v,sink]:
        if tensor.dtype!=torch.bfloat16 or tensor.device.type!='hpu' or tensor.device!=q.device or not tensor.is_contiguous() or tensor.requires_grad:
            return False
    for tensor in [window_block_list,window_block_groups,input_positions]:
        if tensor.dtype not in (torch.int32,torch.int64) or tensor.device!=q.device or not tensor.is_contiguous():
            return False
    return (window_block_list.ndim==1 and 0<window_block_list.numel()<=(4096 if batch else 2147483647)
            and window_block_groups.shape==window_block_list.shape
            and input_positions.ndim in (1,2) and input_positions.numel()==q.shape[0])


def flat_pa_window(*, window_block_list, window_block_groups, input_positions,
                   sliding_window, policy='vendor', **kwargs):
    """Use an explicit FP32 policy for a qualified M1 MiMo SWA128 call.

    Long metadata is accepted by the pinned Lazy bridge, whose native storage
    is I32. The TPC glue rejects any actual I64 tensor; no low-word ABI shortcut
    or silently truncated custom reader is used. Page IDs must fit signed I32.
    fp32_quad is an explicit experimental policy: short context may regress.
    Position is device metadata, so dispatch never reads it back or invents a
    shape-based short-context cutoff. All existing structural guards apply.
    All padded groups must be -1; active request group is zero. Logical order
    must match the production _window_block_tables result.
    """
    if policy not in ('vendor','fp32','fp32_fast','fp32_quad','fp32_av_hoist','fp32_batch','fp32_batch_quad'):
        raise ValueError('SWA policy must be vendor, fp32, fp32_fast, fp32_quad, or fp32_av_hoist')
    if policy!='vendor' and eligible_window_call(kwargs,window_block_list,window_block_groups,input_positions,sliding_window,batch=policy in ('fp32_batch','fp32_batch_quad')):
        op=(torch.ops.gaudi_swa128_batch.window_quad_fp32 if policy=='fp32_batch_quad' and kwargs['query'].shape[0]>1 else
            torch.ops.gaudi_swa128_batch.window_fp32 if policy in ('fp32_batch','fp32_batch_quad') and kwargs['query'].shape[0]>1 else
            torch.ops.gaudi_swa128_avgroup.group if policy in ('fp32_av_hoist','fp32_batch','fp32_batch_quad') else
            torch.ops.gaudi_swa128_ilp.quad if policy=='fp32_quad' else
            torch.ops.gaudi_swa128.forward_window_fast_fp32 if policy=='fp32_fast'
            else torch.ops.gaudi_swa128.forward_window_fp32)
        return op(
            kwargs['query'],kwargs['key_cache'],kwargs['value_cache'],
            window_block_list,window_block_groups,input_positions,kwargs['sinks'],kwargs['scale'])
    from vllm_gaudi.extension import ops
    return ops.flat_pa(**kwargs)
