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


def rms(x, weight, eps):
    f=x.float()
    return (f*torch.rsqrt(f.square().mean(-1,keepdim=True)+eps)*weight.float()).to(x.dtype)


def rope(x, positions, rotary, theta):
    # x: [B,T,H,D]; only the first rotary dimensions participate.
    freq=theta**(-torch.arange(0,rotary,2,dtype=torch.float32,device=x.device)/rotary)
    angle=positions.float()[...,None]*freq
    cosine=torch.cat((angle.cos(),angle.cos()),-1)[:,:,None,:]
    sine=torch.cat((angle.sin(),angle.sin()),-1)[:,:,None,:]
    a=x[...,:rotary].float();half=rotary//2
    rotated=a*cosine+torch.cat((-a[...,half:],a[...,:half]),-1)*sine
    return torch.cat((rotated.to(x.dtype),x[...,rotary:]),-1)


def attention(q,k,v,qpos,kpos,valid,sinks,window,*,fold_heads=False):
    # GQA shares K/V across groups without physically repeating the cache.
    b,t,h,d=q.shape;s=k.shape[1];kh=k.shape[2];g=h//kh
    qg=q.reshape(b,t,kh,g,d).permute(0,2,3,1,4).float()
    kg=k.permute(0,2,1,3).float()
    vg=v.permute(0,2,1,3).float()
    # Keep K/V sharing in the actual MME graph: a group-axis broadcast
    # otherwise produces physical DMA expansions on Gaudi. Dtypes, masks,
    # sink normalization and FP32 MACs are unchanged.
    qm=qg.reshape(b,kh,g*t,d)
    scores=(((qm@kg.transpose(-1,-2))*(d**-.5)).reshape(b,kh,g,t,s)
            if fold_heads else (qg@kg.unsqueeze(2).transpose(-1,-2))*(d**-.5))
    mask=valid[:,None,:]&((qpos[:,:,None]-kpos[:,None,:]).abs()<window)
    scores=scores.masked_fill(~mask[:,None,None,:,:],float('-inf'))
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

    def w(self,key):return self.weight[key.replace('.','__')]

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

    def kv(self,layer,x,positions):
        s=self.spec;p=f'layers.{layer}.self_attn.';shape=(*x.shape[:2],s.kv_heads,s.dim)
        k=self.linear(p+'k_proj.weight',x).reshape(shape)
        k=rope(rms(k,self.w(p+'k_norm.weight'),s.eps),positions,s.rotary,s.theta)
        v=self.linear(p+'v_proj.weight',x).reshape(shape)*s.value_scale
        return k,v

    def project_context(self,features,positions):
        """Project only newly committed target features; never include rejected rows."""
        s=self.spec
        assert features.shape[-1]==len(s.target_layers)*s.hidden
        x=rms(self.linear('fc.weight',features),self.w('hidden_norm.weight'),s.eps)
        return tuple(self.kv(i,x,positions)for i in range(s.layers))

    def noise(self,anchor):
        s=self.spec
        return torch.cat((anchor[:,None,:],self.mask_embedding[None,None,:].expand(anchor.shape[0],s.block-1,s.hidden)),1)

    def forward(self,noise,positions,context,context_positions,context_valid):
        s=self.spec;x=noise
        assert x.shape[1]==s.block and len(context)==s.layers
        kpos=torch.cat((context_positions,positions),1)
        valid=torch.cat((context_valid,torch.ones_like(positions,dtype=torch.bool)),1)
        for i in range(s.layers):
            p=f'layers.{i}.';a=p+'self_attn.'
            z=rms(x,self.w(p+'input_layernorm.weight'),s.eps)
            q=self.linear(a+'q_proj.weight',z).reshape(*z.shape[:2],s.heads,s.dim)
            q=rope(rms(q,self.w(a+'q_norm.weight'),s.eps),positions,s.rotary,s.theta)
            k,v=self.kv(i,z,positions);ck,cv=context[i]
            out=attention(q,torch.cat((ck,k),1),torch.cat((cv,v),1),positions,kpos,valid,
                          self.w(a+'attention_sink_bias'),s.window,fold_heads=getattr(self, 'fold_gqa', False))
            x=x+self.linear(a+'o_proj.weight',out)
            z=rms(x,self.w(p+'post_attention_layernorm.weight'),s.eps)
            gate=self.linear(p+'mlp.gate_proj.weight',z);up=self.linear(p+'mlp.up_proj.weight',z)
            # GPU fused SiLU-and-mul: FP32 activation/product, one BF16 store.
            product=(F.silu(gate.float())*up.float()).to(z.dtype)
            x=x+self.linear(p+'mlp.down_proj.weight',product)
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
