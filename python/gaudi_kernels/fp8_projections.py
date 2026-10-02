"""Load-time fusion for projections sharing the SAME activation tensor.

Ordinary separable-scale FP8 only. This is not a block-scale conversion or an
automatic vLLM hook. Device performance/placement remains unqualified.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedFP8Projections:
    linear: object
    widths: tuple
    format_version: int = 1

    def validate(self):
        from .torch_linear import PreparedSeparableFP8
        if (self.format_version!=1 or not isinstance(self.linear,PreparedSeparableFP8)
                or self.linear.format_version!=1 or not self.widths
                or any(not isinstance(n,int) or n<=0 for n in self.widths)
                or sum(self.widths)!=self.linear.weight.shape[0]):
            raise ValueError('incompatible prepared projection layout')

    def to(self,device):
        self.validate()
        return PreparedFP8Projections(self.linear.to(device),self.widths,self.format_version)


def prepare_fp8_projections(weights,scales,biases=None):
    """Prepare once on CPU: concatenate N, without padding or expanded weights.

    Every projection must consume the same K-wide activation and policy.
    Per-channel native-range adaptation remains identical to separate loading.
    Original CPU checkpoint tensors are not mutated. Transfer only the combined
    owner to HPU; retaining separate device weights would defeat the memory plan.
    """
    import torch
    from .torch_linear import prepare_separable_fp8,PreparedSeparableFP8
    weights,scales=tuple(weights),tuple(scales)
    biases=(None,)*len(weights) if biases is None else tuple(biases)
    if not weights or len(weights)!=len(scales) or len(weights)!=len(biases):
        raise ValueError('matching nonempty projection lists required')
    prepared=tuple(prepare_separable_fp8(w,s,b) for w,s,b in zip(weights,scales,biases))
    if len({p.weight.shape[1] for p in prepared})!=1:
        raise ValueError('fused projections must share K')
    widths=tuple(p.weight.shape[0] for p in prepared)
    # Concatenate bytes, not a dequantized representation.
    weight=torch.cat(tuple(p.weight.view(torch.uint8) for p in prepared),0).view(torch.float8_e4m3fn)
    result=PreparedFP8Projections(PreparedSeparableFP8(weight,
        torch.cat(tuple(p.scale for p in prepared)),torch.cat(tuple(p.bias for p in prepared))),widths)
    result.validate()
    assert result.linear.weight.numel()==sum(w.numel() for w in weights)
    return result


def linear_fp8_projections(x,prepared,*,activation,optimized=False,quantization=None):
    """Return one [M,sum(N)] tensor in the current graph, using one Linear.

    This constructs one quant/MME/epilogue chain. It does not by itself prove
    a single hardware launch or faster runtime; a captured-graph audit must.
    """
    from .torch_linear import linear_fp8
    if not isinstance(prepared,PreparedFP8Projections):raise ValueError('prepared projections required')
    prepared.validate()
    return linear_fp8(x,prepared.linear,activation=activation,optimized=optimized,quantization=quantization)


def split_fp8_projections(output,prepared,*,contiguous=False):
    """Explicit output views; contiguous=True may add materialization nodes.

    M>1 views normally retain the combined row stride. A consumer requiring
    contiguous Q/K/V can erase the fusion benefit; qualify that whole graph.
    """
    prepared.validate()
    if output.ndim!=2 or output.shape[1]!=sum(prepared.widths):
        raise ValueError('combined output shape mismatch')
    pieces=output.split(prepared.widths,dim=1)
    return tuple(piece.contiguous() for piece in pieces) if contiguous else pieces
