"""Teacher-forced logit diagnostics; never imported by production policy."""

def compare_logits(reference, candidate, *, contract='legacy'):
    import torch
    if contract not in ('legacy', 'fp32_arithmetic_v1'):
        raise ValueError('Unknown quality arithmetic contract')
    if reference.shape != candidate.shape:
        raise ValueError('Teacher logits shape mismatch')
    (reference, candidate) = (reference.double(), candidate.double())
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    if not finite:
        return {'pass': False, 'finite': False}
    (lp, lq) = (reference.log_softmax(-1), candidate.log_softmax(-1))
    kl = float((lp.exp() * (lp - lq)).sum(-1).max())
    relative_l2 = float((reference - candidate).norm() / reference.norm().clamp_min(1e-30))
    # FP32 accumulation permits a different addition tree; it does not exempt
    # full-model logits from the quality limit. Check every vocabulary row so
    # one bad position cannot be hidden by an aggregate norm over the batch.
    row_l2 = (reference-candidate).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-30)
    max_relative_l2 = float(row_l2.max())
    return {'pass': kl <= 0.01 and relative_l2 <= 0.02 and max_relative_l2 <= 0.02,
            'finite': True, 'max_kl_ref_to_candidate': kl, 'relative_l2': relative_l2,
            'max_row_relative_l2': max_relative_l2,
            'top1_match_fraction': float((reference.argmax(-1) == candidate.argmax(-1)).double().mean()),
            'gate': {'max_kl': 0.01, 'relative_l2': 0.02, 'max_row_relative_l2': 0.02},
            'scope': 'full-vocabulary teacher-forced logits; bounded regression gate, not broad quality evaluation',
            'contract': contract}
