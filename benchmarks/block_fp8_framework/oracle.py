"""Independent complete CPU MAC references, never used by the HPU apply path."""
import torch


def original_weight(raw, scales):
    n,k=raw.shape
    s=scales.repeat_interleave(128,0).repeat_interleave(128,1)[:n,:k]
    return raw.double()*s.double()


def adapted_weight(prepared, scale_math='fp32'):
    w=prepared.weight.permute(1,0,2).contiguous().reshape(prepared.n,-1)[:,:prepared.k].float()
    s=prepared.scales.repeat_interleave(128,0).repeat_interleave(128,1)[:prepared.n,:prepared.k]
    if scale_math=='bf16':s=s.bfloat16().float()
    return (w*s).bfloat16()


def quantize_block(x):
    """Original OCP RNE followed by native half RNE; exact contractual FP32 scale."""
    m,k=x.shape;g=(k+127)//128
    padded=torch.zeros((m,g*128),dtype=torch.float32);padded[:,:k]=x.float()
    blocks=padded.reshape(m,g,128).permute(1,0,2).contiguous()
    scale=blocks.abs().amax(-1,keepdim=True).clamp_min(1e-10)*torch.tensor(1/448,dtype=torch.float32)
    ocp=(blocks/scale).clamp(-448,448).to(torch.float8_e4m3fn)
    native=(ocp.float()*.5).to(torch.float8_e4m3fn)
    return native,scale*2,ocp,scale


def reference(x, raw, scales, bias, prepared, activation, scale_math='fp32'):
    high=x.double()@original_weight(raw,scales).T+bias.double()
    if activation=='bf16':
        adapted=x.double()@adapted_weight(prepared,scale_math).double().T+bias.double()
        gpu=high
    else:
        q,sa,ocp,original_sa=quantize_block(x)
        g=q.shape[0];m=x.shape[0];n=raw.shape[0]
        padded=torch.zeros((n,g*128),dtype=torch.float64);padded[:,:raw.shape[1]]=raw.double()
        blocks=padded.reshape(n,g,128).permute(1,0,2)
        adapted=torch.zeros((m,n),dtype=torch.float64);gpu=adapted.clone()
        for i in range(g):
            ws=prepared.scales[:,i].repeat_interleave(128)[:n]
            s0=scales[:,i].repeat_interleave(128)[:n]
            adapted+=(q[i].double()@prepared.weight[i].double().T)*(sa[i].double()*ws.double()[None,:])
            gpu+=(ocp[i].double()@blocks[i].T)*(original_sa[i].double()*s0.double()[None,:])
        adapted+=bias.double();gpu+=bias.double()
    return {'high_fp64':high,'adapted_fp64':adapted,'gpu_style_fp64':gpu}


def errors(actual, refs):
    a=actual.double();result={}
    for name,ref in refs.items():
        d=a-ref
        result[name]={'relative_l2':float(d.norm()/ref.norm().clamp_min(1e-30)),
                     'max_abs':float(d.abs().max()),'rms':float(d.square().mean().sqrt())}
    result['finite']=bool(torch.isfinite(a).all())
    return result
