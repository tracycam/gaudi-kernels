# SPDX-License-Identifier: Apache-2.0
"""MiMo DFlash checkpoint reference, before TP/native-serving integration.

BF16 stored projections; FP32 norm and attention accumulation. The deliberately
explicit attention is a correctness reference, not the performance backend.
Matches the MiMo extras in SGLang 98fce73d5b: V scaling, sink, partial Neox RoPE,
and symmetric noncausal SWA (window-1 on both sides). No target weight mutation.
"""
from dataclasses import dataclass,replace
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Spec:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    dim: int
    rotary: int
    window: int
    block: int
    eps: float
    theta: float
    value_scale: float
    target_layers: tuple
    mask_id: int

    @classmethod
    def parse(cls, c):
        d=c['dflash_config'];dim=c['head_dim'];rotary=int(dim*c['partial_rotary_factor'])
        assert c['is_causal']is False and c['attention_bias']is False
        assert c['hidden_act']=='silu' and c['v_head_dim']==dim
        assert c['layer_types']==['sliding_attention']*c['num_hidden_layers']
        assert d['attention_sink_bias']is True and not c.get('rope_scaling')
        assert c['hidden_size']==c['target_hidden_size'] and c['block_size']==d['block_size']
        assert 0<rotary<=dim and rotary%2==0 and c['num_attention_heads']%c['num_key_value_heads']==0
        return cls(c['hidden_size'],c['intermediate_size'],c['num_hidden_layers'],c['num_attention_heads'],
                   c['num_key_value_heads'],dim,rotary,c['sliding_window'],c['block_size'],c['rms_norm_eps'],
                   c['rope_theta'],d['attention_value_scale'],tuple(d['target_layer_ids']),d['mask_token_id'])


def rms(x, weight, eps, residual=None):
    if residual is not None:
        if (x.device.type=='hpu' and x.dtype==torch.bfloat16 and residual.dtype==torch.bfloat16
                and weight.dtype==torch.bfloat16 and x.is_contiguous() and residual.is_contiguous()
                and weight.is_contiguous() and x.shape==residual.shape and 0<x.shape[-1]<=8192
                and hasattr(torch.ops,'gaudi_kernels')
                and hasattr(torch.ops.gaudi_kernels,'_residual_rmsnorm_bf16')):
            h=x.shape[-1]
            reshape_op=torch.ops.gaudi_kernels._norm_reshape
            mx=x if x.ndim==2 else reshape_op(x,[x.numel()//h,h])
            mr=residual if residual.ndim==2 else reshape_op(residual,[residual.numel()//h,h])
            ro,no=torch.ops.gaudi_kernels._residual_rmsnorm_bf16(mx,mr,weight,float(eps))
            if x.ndim!=2:
                no=reshape_op(no,list(x.shape))
                ro=reshape_op(ro,list(residual.shape))
            return no,ro
        rf=x.float()+residual.float()
        ro=rf.to(x.dtype)
        no=(rf*torch.rsqrt(rf.square().mean(-1,keepdim=True)+eps)*weight.float()).to(x.dtype)
        return no,ro
    if (x.device.type=='hpu' and x.dtype==torch.bfloat16 and weight.dtype==torch.bfloat16
            and x.is_contiguous() and weight.is_contiguous() and 0<x.shape[-1]<=8192
            and hasattr(torch.ops,'gaudi_kernels')
            and hasattr(torch.ops.gaudi_kernels,'_pure_rmsnorm_bf16')):
        h=x.shape[-1]
        reshape_op=torch.ops.gaudi_kernels._norm_reshape
        mx=x if x.ndim==2 else reshape_op(x,[x.numel()//h,h])
        no=torch.ops.gaudi_kernels._pure_rmsnorm_bf16(mx,weight,float(eps))
        return no if x.ndim==2 else reshape_op(no,list(x.shape))
    f=x.float()
    return (f*torch.rsqrt(f.square().mean(-1,keepdim=True)+eps)*weight.float()).to(x.dtype)


try:
    from habana_frameworks.torch.hpex.kernels import (
        RotaryPosEmbeddingMode as _RotaryPosEmbeddingMode,
        apply_rotary_pos_emb as _apply_rotary_pos_emb,
    )
    _ROPE_BLOCKWISE = _RotaryPosEmbeddingMode.BLOCKWISE
except Exception:
    _apply_rotary_pos_emb = None
    _ROPE_BLOCKWISE = None


def _cos_sin(positions, rotary, theta, inv_freq=None):
    if inv_freq is None or inv_freq.device != positions.device:
        half = theta**(-torch.arange(0,rotary,2,dtype=torch.float32,device=positions.device)/rotary)
        inv_freq = torch.cat((half,half),-1)
    angle = positions.float()[...,None]*inv_freq
    return angle.cos()[:,:,None,:], angle.sin()[:,:,None,:]


def rope(x, positions, rotary, theta, *, cos_sin=None):
    # x: [B,T,H,D]; only the first rotary dimensions participate.
    cosine,sine = _cos_sin(positions,rotary,theta) if cos_sin is None else cos_sin
    if x.device.type=='hpu' and _apply_rotary_pos_emb is not None:
        rotated = _apply_rotary_pos_emb(x[...,:rotary].float(),cosine,sine,None,0,_ROPE_BLOCKWISE).to(x.dtype)
        return torch.cat((rotated,x[...,rotary:]),-1)
    a=x[...,:rotary].float();half=rotary//2
    rotated=a*cosine+torch.cat((-a[...,half:],a[...,:half]),-1)*sine
    return torch.cat((rotated.to(x.dtype),x[...,rotary:]),-1)


def attention(q,k,v,qpos,kpos,valid,sinks,window,*,fold_heads=False,neg_mask=None,sink=None):
    # GQA shares K/V across groups without physically repeating the cache.
    b,t,h,d=q.shape;s=k.shape[1];kh=k.shape[2];g=h//kh
    qg=q.reshape(b,t,kh,g,d).permute(0,2,3,1,4).float()
    kg=k.permute(0,2,1,3).float()
    vg=v.permute(0,2,1,3).float()
    # Optional fold removes group-axis broadcast operands. Compiler tiling,
    # physical traffic and latency are measured separately; sharing a source
    # tensor does not by itself prove a bandwidth reduction.
    if fold_heads:
        qm=qg.reshape(b,kh,g*t,d)
        scores=((qm@kg.transpose(-1,-2))*(d**-.5)).reshape(b,kh,g,t,s)
    else:
        scores=(qg@kg.unsqueeze(2).transpose(-1,-2))*(d**-.5)
    if neg_mask is None:
        mask=valid[:,None,:]&((qpos[:,:,None]-kpos[:,None,:]).abs()<window)
        neg_mask=~mask[:,None,None,:,:]
    scores=scores.masked_fill(neg_mask,float('-inf'))
    if sink is None:
        sink=sinks.float().reshape(1,kh,g,1,1).expand(b,kh,g,t,1)
    # The sink contributes to the denominator but has zero V. Including it
    # keeps an all-masked KV row finite without deleting the sink semantics.
    probs=torch.cat((scores,sink),-1).softmax(-1)[...,:s]
    out=((probs.reshape(b,kh,g*t,s)@vg).reshape(b,kh,g,t,d)
         if fold_heads else probs@vg.unsqueeze(2))
    return out.permute(0,3,1,2,4).reshape(b,t,h*d).to(q.dtype)


def partition(key,shape,rank,size):
    """Checkpoint axes, with no re-quantization or permanent dtype expansion."""
    assert 0<=rank<size
    axis=None
    if key.endswith(('.o_proj.weight','.down_proj.weight')):axis=1
    elif key=='fc.weight':axis=0
    elif key.endswith(('.q_proj.weight','.k_proj.weight','.v_proj.weight','.gate_proj.weight','.up_proj.weight','.attention_sink_bias')):axis=0
    slices=[slice(None)]*len(shape)
    if axis is not None:
        assert shape[axis]%size==0,(key,shape,size)
        width=shape[axis]//size;slices[axis]=slice(rank*width,(rank+1)*width)
    return tuple(slices)


class Draft(nn.Module):
    def __init__(self,spec,weights,mask_embedding,*,tp_rank=0,tp_size=1,reduce_sum=None,gather_output=None):
        super().__init__();self.spec=spec
        assert tp_size==1 or (callable(reduce_sum)and callable(gather_output))
        self.tp_rank,self.tp_size,self.reduce_sum=tp_rank,tp_size,reduce_sum
        self.gather_output=gather_output
        self.weight=nn.ParameterDict({k.replace('.','__'):nn.Parameter(v,requires_grad=False)for k,v in weights.items()})
        self.register_buffer('mask_embedding',mask_embedding.reshape(spec.hidden))
        half_freq=spec.theta**(-torch.arange(0,spec.rotary,2,dtype=torch.float32,device=self.mask_embedding.device)/spec.rotary)
        self.register_buffer('_inv_freq',torch.cat((half_freq,half_freq),-1),persistent=False)
        self._hpu_cache=None
        if self.mask_embedding.device.type=='hpu':
            self._ensure_hpu_cache()

    def w(self,key):return self.weight[key.replace('.','__')]

    def _ensure_hpu_cache(self):
        s=self.spec;dev=self.mask_embedding.device
        if dev.type!='hpu':
            return None
        sig=(dev,self.mask_embedding._cdata,tuple(p._cdata for p in self.weight.values()))
        if self._hpu_cache is not None and self._hpu_cache[0]==sig:
            return self._hpu_cache[1]
        kh=s.kv_heads;g=s.heads//kh
        w_all_k=torch.cat([self.w(f'layers.{i}.self_attn.k_proj.weight')for i in range(s.layers)],dim=0)
        w_all_v=torch.cat([self.w(f'layers.{i}.self_attn.v_proj.weight')for i in range(s.layers)],dim=0)
        w_all_kv=torch.cat((w_all_k,w_all_v),dim=0)
        k_norms_f32=torch.stack([self.w(f'layers.{i}.self_attn.k_norm.weight').float()for i in range(s.layers)],dim=0)[None,None,:,None,:].contiguous()
        w_qkv=tuple(
            torch.cat((
                self.w(f'layers.{i}.self_attn.q_proj.weight'),
                self.w(f'layers.{i}.self_attn.k_proj.weight'),
                self.w(f'layers.{i}.self_attn.v_proj.weight'),
            ),dim=0)
            for i in range(s.layers)
        )
        qk_norms_f32=tuple(
            torch.cat((
                self.w(f'layers.{i}.self_attn.q_norm.weight').float().unsqueeze(0).expand(s.heads,s.dim),
                self.w(f'layers.{i}.self_attn.k_norm.weight').float().unsqueeze(0).expand(kh,s.dim),
            ),dim=0)[None,None,:,:].contiguous()
            for i in range(s.layers)
        )
        w_gate_up=tuple(
            torch.cat((
                self.w(f'layers.{i}.mlp.gate_proj.weight'),
                self.w(f'layers.{i}.mlp.up_proj.weight'),
            ),dim=0)
            for i in range(s.layers)
        )
        sinks_f32_block=tuple(
            self.w(f'layers.{i}.self_attn.attention_sink_bias').float().reshape(1,kh,g,1,1).expand(1,kh,g,s.block,1).contiguous()
            for i in range(s.layers)
        )
        mask_emb_block=self.mask_embedding[None,None,:].expand(1,s.block-1,s.hidden).contiguous()
        ones_pos_bool=torch.ones((1,s.block),dtype=torch.bool,device=dev)
        cache={
            'w_all_kv':w_all_kv,
            'k_norms_f32':k_norms_f32,
            'w_qkv':w_qkv,
            'qk_norms_f32':qk_norms_f32,
            'w_gate_up':w_gate_up,
            'sinks_f32_block':sinks_f32_block,
            'mask_emb_block':mask_emb_block,
            'ones_pos_bool':ones_pos_bool,
        }
        self._hpu_cache=(sig,cache)
        return cache

    def linear(self,key,x):
        out=F.linear(x,self.w(key))
        if self.tp_size>1 and key=='fc.weight':
            # SGLang MiMo fc is a replicated dense projection. Output sharding
            # retains full K and adds no BF16 partial-sum rounding.
            out=self.gather_output(out)
        if self.tp_size>1 and key.endswith(('.o_proj.weight','.down_proj.weight')):
            # Conventional BF16 row-parallel linear: FP32 local MAC, BF16
            # partial result, explicit FP32 collective, BF16 layer boundary.
            out=self.reduce_sum(out.float()).to(x.dtype)
        return out

    def _reshape(self,x,shape):
        if (x.device.type=='hpu' and x.dtype==torch.bfloat16 and x.is_contiguous()
                and hasattr(torch.ops,'gaudi_kernels') and hasattr(torch.ops.gaudi_kernels,'_norm_reshape')):
            return torch.ops.gaudi_kernels._norm_reshape(x,list(shape))
        return x.reshape(shape)

    def kv(self,layer,x,positions,*,cos_sin=None):
        s=self.spec;p=f'layers.{layer}.self_attn.';shape=(*x.shape[:2],s.kv_heads,s.dim)
        k=self._reshape(self.linear(p+'k_proj.weight',x),shape)
        k=rope(rms(k,self.w(p+'k_norm.weight'),s.eps),positions,s.rotary,s.theta,cos_sin=cos_sin)
        v=self.linear(p+'v_proj.weight',x).reshape(shape)
        if s.value_scale!=1.0:
            v=v*s.value_scale
        return k,v

    def project_context(self,features,positions):
        """Project only newly committed target features; never include rejected rows."""
        s=self.spec
        assert features.shape[-1]==len(s.target_layers)*s.hidden
        fc_out=self.linear('fc.weight',features)
        if self.tp_size>1 and fc_out.device.type=='hpu':
            f=fc_out.float()
            x=(f*torch.rsqrt(f.square().mean(-1,keepdim=True)+s.eps)*self.w('hidden_norm.weight').float()).to(fc_out.dtype)
        else:
            x=rms(fc_out,self.w('hidden_norm.weight'),s.eps)
        if features.device.type=='hpu' and 'linear'not in self.__dict__ and 'kv'not in self.__dict__:
            cache=self._ensure_hpu_cache()
            b,t_ctx=x.shape[:2]
            kv_all=F.linear(x,cache['w_all_kv']).reshape(b,t_ctx,2,s.layers,s.kv_heads,s.dim)
            k_all,v_all=kv_all[:,:,0],kv_all[:,:,1]
            if s.value_scale!=1.0:
                v_all=v_all*s.value_scale
            kf=k_all.float()
            k_normed=(kf*torch.rsqrt(kf.square().mean(-1,keepdim=True)+s.eps)*cache['k_norms_f32']).to(x.dtype)
            cos_sin=_cos_sin(positions,s.rotary,s.theta,self._inv_freq)
            k_rot=rope(k_normed.reshape(b,t_ctx,s.layers*s.kv_heads,s.dim),positions,s.rotary,s.theta,cos_sin=cos_sin).reshape(b,t_ctx,s.layers,s.kv_heads,s.dim)
            return tuple((k_rot[:,:,i],v_all[:,:,i])for i in range(s.layers))
        cos_sin=_cos_sin(positions,s.rotary,s.theta,self._inv_freq)
        return tuple(self.kv(i,x,positions,cos_sin=cos_sin)for i in range(s.layers))

    def noise(self,anchor):
        s=self.spec;b=anchor.shape[0]
        if anchor.device.type=='hpu':
            cache=self._ensure_hpu_cache()
            mask_emb=cache['mask_emb_block']if b==1 else cache['mask_emb_block'].expand(b,s.block-1,s.hidden)
            return torch.cat((anchor[:,None,:],mask_emb),1)
        return torch.cat((anchor[:,None,:],self.mask_embedding[None,None,:].expand(b,s.block-1,s.hidden)),1)

    def forward(self,noise,positions,context,context_positions,context_valid):
        s=self.spec;x=noise;residual=None
        assert x.shape[1]==s.block and len(context)==s.layers
        b,t=x.shape[:2];kh=s.kv_heads;g=s.heads//kh;d=s.dim
        kpos=torch.cat((context_positions,positions),1)
        use_hpu_fast=x.device.type=='hpu' and 'linear'not in self.__dict__ and 'kv'not in self.__dict__
        cache=self._ensure_hpu_cache()if use_hpu_fast else None
        ones_pos=(cache['ones_pos_bool']if b==1 else cache['ones_pos_bool'].expand(b,t))if cache is not None else torch.ones_like(positions,dtype=torch.bool)
        valid=torch.cat((context_valid,ones_pos),1)
        mask=valid[:,None,:]&((positions[:,:,None]-kpos[:,None,:]).abs()<s.window)
        neg_mask=~mask[:,None,None,:,:]
        cos_sin=_cos_sin(positions,s.rotary,s.theta,self._inv_freq)
        fold_heads=getattr(self,'fold_gqa',False)
        for i in range(s.layers):
            p=f'layers.{i}.';a=p+'self_attn.'
            if residual is None:
                z=rms(x,self.w(p+'input_layernorm.weight'),s.eps)
            else:
                z,x=rms(x,self.w(p+'input_layernorm.weight'),s.eps,residual=residual)
            if cache is not None:
                qkv=F.linear(z,cache['w_qkv'][i]).reshape(b,t,s.heads+2*kh,d)
                qk=qkv[:,:,:s.heads+kh];v=qkv[:,:,s.heads+kh:]
                if s.value_scale!=1.0:
                    v=v*s.value_scale
                qkf=qk.float()
                qk_normed=(qkf*torch.rsqrt(qkf.square().mean(-1,keepdim=True)+s.eps)*cache['qk_norms_f32'][i]).to(z.dtype)
                qk_rot=rope(qk_normed,positions,s.rotary,s.theta,cos_sin=cos_sin)
                q=qk_rot[:,:,:s.heads];k=qk_rot[:,:,s.heads:]
                sink=cache['sinks_f32_block'][i]if (b==1 and t==s.block)else cache['sinks_f32_block'][i][:,:,:,:1,:].expand(b,kh,g,t,1)
            else:
                q=self._reshape(self.linear(a+'q_proj.weight',z),(*z.shape[:2],s.heads,s.dim))
                q=rope(rms(q,self.w(a+'q_norm.weight'),s.eps),positions,s.rotary,s.theta,cos_sin=cos_sin)
                k,v=self.kv(i,z,positions,cos_sin=cos_sin)
                sink=None
            ck,cv=context[i]
            out=attention(q,torch.cat((ck,k),1),torch.cat((cv,v),1),positions,kpos,valid,
                          self.w(a+'attention_sink_bias'),s.window,fold_heads=fold_heads,
                          neg_mask=neg_mask,sink=sink)
            z,x=rms(self.linear(a+'o_proj.weight',out),self.w(p+'post_attention_layernorm.weight'),s.eps,residual=x)
            if cache is not None:
                gate,up=F.linear(z,cache['w_gate_up'][i]).split(s.intermediate,dim=-1)
            else:
                gate=self.linear(p+'mlp.gate_proj.weight',z);up=self.linear(p+'mlp.up_proj.weight',z)
            # GPU fused SiLU-and-mul: FP32 activation/product, one BF16 store.
            product=(F.silu(gate.float())*up.float()).to(z.dtype)
            residual=x
            x=self.linear(p+'mlp.down_proj.weight',product)
        if residual is not None:
            z,_=rms(x,self.w('norm.weight'),s.eps,residual=residual)
            return z
        return rms(x,self.w('norm.weight'),s.eps)


def load_checkpoint(directory,device='cpu',*,tp_rank=0,tp_size=1,reduce_sum=None,gather_output=None):
    from safetensors import safe_open
    root=Path(directory);spec=Spec.parse(json.loads((root/'config.json').read_text()))
    assert tp_size>=1 and 0<=tp_rank<tp_size
    assert all(n%tp_size==0 for n in (spec.heads,spec.kv_heads,spec.intermediate,len(spec.target_layers)*spec.hidden))
    assert tp_size==1 or (callable(reduce_sum)and callable(gather_output))
    expected={'fc.weight':(spec.hidden,len(spec.target_layers)*spec.hidden),'hidden_norm.weight':(spec.hidden,),
              'norm.weight':(spec.hidden,)}
    for i in range(spec.layers):
        p=f'layers.{i}.';a=p+'self_attn.'
        for n in ('input_layernorm.weight','post_attention_layernorm.weight'):expected[p+n]=(spec.hidden,)
        for n in ('gate_proj.weight','up_proj.weight'):expected[p+'mlp.'+n]=(spec.intermediate,spec.hidden)
        expected[p+'mlp.down_proj.weight']=(spec.hidden,spec.intermediate)
        expected[a+'q_proj.weight']=(spec.heads*spec.dim,spec.hidden)
        for n in ('k_proj.weight','v_proj.weight'):expected[a+n]=(spec.kv_heads*spec.dim,spec.hidden)
        expected[a+'o_proj.weight']=(spec.hidden,spec.heads*spec.dim)
        for n in ('q_norm.weight','k_norm.weight'):expected[a+n]=(spec.dim,)
        expected[a+'attention_sink_bias']=(spec.heads,)
    with safe_open(root/'dflash_draft_model.safetensors',framework='pt',device='cpu')as f:
        assert set(f.keys())==set(expected),'unknown or missing checkpoint tensors'
        weights={}
        for key,shape in expected.items():
            view=f.get_slice(key);assert tuple(view.get_shape())==shape,(key,view.get_shape())
            value=view[partition(key,shape,tp_rank,tp_size)].contiguous()
            assert value.dtype==torch.bfloat16,(key,value.dtype)
            weights[key]=value.to(device)
    mask=torch.load(root/'mask_embedding.pt',map_location='cpu',weights_only=True)
    assert int(mask['mask_token_id'])==spec.mask_id and not mask.get('per_position')
    value=mask['embedding'];assert value.numel()==spec.hidden and torch.isfinite(value).all()
    local=replace(spec,heads=spec.heads//tp_size,kv_heads=spec.kv_heads//tp_size,intermediate=spec.intermediate//tp_size)
    return Draft(local,weights,value.to(device=device,dtype=torch.bfloat16),tp_rank=tp_rank,tp_size=tp_size,reduce_sum=reduce_sum,gather_output=gather_output)
