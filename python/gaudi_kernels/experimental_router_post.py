"""Opt-in post-vendor-TopK experiment. Import changes no router or policy."""
import math
import torch


def post_top8(scores, ordered_ids, renormalize, factor, *, variant, diagnostics=False):
    """Preserve supplied slot order; gather original FP32 scores only.

    Domain is the M1/E384/top8 sigmoid or softmax probability path. Zero and
    subnormal handling follows native FP32 FTZ arithmetic, not an FP64 promise.
    Denominator zero is not clamped. The diagnostic output is the actual sum.
    """
    if variant not in ('scalar', 'vector'):
        raise ValueError('explicit scalar/vector candidate required')
    if (scores.shape != (1,384) or ordered_ids.shape != (1,8)
            or scores.dtype != torch.float32 or ordered_ids.dtype not in (torch.int32,torch.int64)):
        raise ValueError('FP32 scores[1,384] and ordered integer IDs[1,8] required')
    if (scores.device.type != 'hpu' or scores.device != ordered_ids.device
            or not scores.is_contiguous() or not ordered_ids.is_contiguous()
            or scores.requires_grad):
        raise ValueError('contiguous HPU inference producer tensors required')
    if not math.isfinite(factor) or abs(factor) > 3.4028234663852886e38:
        raise ValueError('finite FP32 factor required')
    # Preserve the existing explicit int32 conversion. Do not infer actual
    # storage type from the legacy bridge's logical-Long TopK metadata.
    ids = ordered_ids if ordered_ids.dtype == torch.int32 else ordered_ids.to(torch.int32)
    w,i,s = getattr(torch.ops.gaudi_router_post8,variant)(scores,ids,float(factor),bool(renormalize))
    return (w,i,s) if diagnostics else (w,i)


def make_grouped_topk_post(original, *, variant):
    """Return a local adapter; caller explicitly installs it before capture.

    Unsupported configurations call the original function unchanged. Vendor
    sigmoid/softmax, correction-bias selection and sorted=False TopK stay intact.
    This factory does not patch a class/module, read device values, or synchronize.
    """
    if variant not in ('scalar','vector'):
        raise ValueError('unknown post-TopK variant')
    def dispatch(gating_output,topk,renormalize,num_expert_group,topk_group,
                 scoring_func,routed_scaling_factor,e_score_correction_bias):
        args=(gating_output,topk,renormalize,num_expert_group,topk_group,
              scoring_func,routed_scaling_factor,e_score_correction_bias)
        eligible=(num_expert_group==1 and topk_group==1 and topk==8
                  and gating_output.shape==(1,384) and gating_output.device.type=='hpu'
                  and scoring_func in ('sigmoid','softmax') and not gating_output.requires_grad
                  and math.isfinite(routed_scaling_factor)
                  and abs(routed_scaling_factor)<=3.4028234663852886e38)
        if not eligible:return original(*args)
        scores=gating_output.float()
        scores=scores.sigmoid() if scoring_func=='sigmoid' else torch.softmax(scores,dim=-1)
        selection=scores if e_score_correction_bias is None else scores+e_score_correction_bias.unsqueeze(0)
        # Selection and returned order both remain the vendor's responsibility.
        ids=torch.topk(selection,k=topk,dim=-1,sorted=False)[1]
        return post_top8(scores,ids,renormalize,routed_scaling_factor,variant=variant)
    return dispatch
