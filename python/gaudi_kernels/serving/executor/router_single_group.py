"""Drop only redundant group selection when the sole group is always selected."""
from gaudi_kernels.engine.context import context as execution_context
import os
import torch
_POST = None

def grouped_topk_one(gating_output, topk, renormalize, num_expert_group, topk_group, scoring_func, routed_scaling_factor, e_score_correction_bias):
    assert num_expert_group == 1 and topk_group == 1
    gating_output = gating_output.float()
    if scoring_func == 'sigmoid':
        scores = gating_output.sigmoid()
    elif scoring_func == 'softmax':
        scores = torch.softmax(gating_output, dim=-1)
    else:
        raise ValueError(scoring_func)
    original_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias.unsqueeze(0)
        ids = torch.topk(scores, k=topk, dim=-1, sorted=False)[1]
        weights = None
    else:
        (weights, ids) = torch.topk(scores, k=topk, dim=-1, sorted=False)
    if _POST is not None:
        candidate = _POST(original_scores, ids, renormalize, routed_scaling_factor)
        if candidate is not None:
            return candidate
    if weights is None:
        weights = original_scores.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        weights = weights * routed_scaling_factor
    return (weights, ids.to(torch.int32))

def install():
    global _POST
    if execution_context().has('router_post'):
        from gaudi_kernels.serving.executor.router_post_runtime import prepare, try_post
        prepare()
        _POST = try_post
    from vllm.model_executor.layers.fused_moe.router import cpu_router
    original = cpu_router._grouped_topk

    def dispatch(*args, **kwargs):
        ng = kwargs.get('num_expert_group', args[3] if len(args) > 3 else None)
        kg = kwargs.get('topk_group', args[4] if len(args) > 4 else None)
        logits = kwargs.get('gating_output', args[0] if args else None)
        if ng == 1 and kg == 1 and (logits.shape[0] == 1):
            return grouped_topk_one(*args, **kwargs)
        return original(*args, **kwargs)
    cpu_router._grouped_topk = dispatch
