"""Public graph-owned shape restoration for the pinned MiMo partial RoPE path.

The vendor computation is preserved. Only the final Tensor.reshape after each
cat is expressed as a graph node before the temporary vendor producer expires.
"""
import functools
import torch

_ORIGINAL=None
_WRAPPER=None
_COUNTS={'owned_restore_calls':0,'vendor_calls':0}


def install_rotary_boundary(policy):
    global _ORIGINAL,_WRAPPER
    from vllm_gaudi.ops.hpu_rotary_embedding import HPURotaryEmbedding
    if _ORIGINAL is not None:return
    original=HPURotaryEmbedding.forward_oot

    @functools.wraps(original)
    def forward(self,positions,query,key,offsets=None):
        batch=policy() in ('fp32_batch','fp32_batch_quad')
        rows=positions.numel()if isinstance(positions,torch.Tensor)else 0
        eligible=(policy()!='vendor' and type(self) is HPURotaryEmbedding
                  and self.head_size==192 and self.rotary_dim==64
                  and isinstance(query,torch.Tensor) and tuple(query.shape) in ((rows,3072),(rows,1,3072))
                  and query.dtype==torch.bfloat16 and query.device.type=='hpu'
                  and (query.is_contiguous()or batch) and not query.requires_grad
                  and isinstance(positions,torch.Tensor) and 1<=rows<=(32 if batch else 1)
                  and positions.device==query.device
                  and (key is None or (isinstance(key,torch.Tensor) and key.numel()==rows*192
                       and key.dtype==torch.bfloat16 and key.device==query.device
                       and (key.is_contiguous()or batch) and not key.requires_grad)))
        if not eligible:
            _COUNTS['vendor_calls']+=1
            return original(self,positions,query,key,offsets)
        _COUNTS['owned_restore_calls']+=1
        from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingMode,apply_rotary_pos_emb
        # Identical preparation and vendor RoPE arithmetic to forward_oot.
        if not hasattr(self,'sin') or self.recompute_cos_sin:
            self.prepare_cos_sin(positions,offsets,recompute_cos_sin=True)
        if hasattr(self,'scaling_factors') or hasattr(self,'scaling_factor') or self.sin is None:
            self.prepare_cos_sin(positions,offsets)
        mode=RotaryPosEmbeddingMode.BLOCKWISE if self.is_neox_style else RotaryPosEmbeddingMode.PAIRWISE
        sin,cos=self.sin,self.cos;query_shape=query.shape
        query=query.view(positions.numel(),-1,self.head_size)
        key_shape=key.shape if key is not None else None
        if key is not None:key=key.view(positions.numel(),-1,self.head_size)
        qr=apply_rotary_pos_emb(query[...,:self.rotary_dim],cos,sin,None,0,mode)
        qcat=torch.cat((qr,query[...,self.rotary_dim:]),dim=-1)
        qout=torch.ops.gaudi_swa128.reshape(qcat,list(query_shape))
        if key is None:return qout,None
        kr=apply_rotary_pos_emb(key[...,:self.rotary_dim],cos,sin,None,0,mode)
        kcat=torch.cat((kr,key[...,self.rotary_dim:]),dim=-1)
        kout=torch.ops.gaudi_swa128.reshape(kcat,list(key_shape))
        return qout,kout

    HPURotaryEmbedding.forward_oot=forward;_ORIGINAL=original;_WRAPPER=forward


def prepare_rotary_boundary(model):
    from vllm_gaudi.ops.hpu_rotary_embedding import HPURotaryEmbedding
    matched=rebound=0
    for module in model.modules():
        if type(module) is not HPURotaryEmbedding or module.head_size!=192 or module.rotary_dim!=64:
            continue
        matched+=1
        method=getattr(module,'_forward_method',None)
        if _ORIGINAL is not None and getattr(method,'__func__',None) is _ORIGINAL:
            module._forward_method=_WRAPPER.__get__(module,type(module));rebound+=1
    return {'matched':matched,'dispatch_rebound':rebound,
            'scope':'MiMo M1 partial RoPE restore is conditional on active SWA policy; arithmetic remains vendor'}


def rotary_boundary_snapshot(reset=False):
    result={'python_calls':dict(_COUNTS),'scope':'capture/forward calls, not device replay counts'}
    if reset:
        for name in _COUNTS:_COUNTS[name]=0
    return result
