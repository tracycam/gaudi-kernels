# SPDX-License-Identifier: Apache-2.0
"""Greedy acceptance for packed [anchor, drafts] target queries, on device.

No stochastic approximation. The last emitted token is the next uncommitted
anchor; committed input rows and emitted tokens are separate tensor roles even
when their counts happen to match in this greedy protocol.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class GreedyVerification:
    emitted_ids: torch.Tensor
    committed_input_counts: torch.Tensor
    emitted_counts: torch.Tensor
    matched_draft_counts: torch.Tensor
    valid: torch.Tensor


def verify_ids(input_ids, target_ids, query_lengths, remaining, *, eos_ids=()):
    """Fixed query extents, mutable tensor values; no item()/CPU readback.

    Each request must contain one anchor plus its actual valid drafts. The
    inputs are packed without request padding. Outputs are padded only for
    delivery, after every query's computation. Invalid negative budgets yield
    valid=False and no output/commit, rather than an implicit clamp/promotion.
    """
    if (type(query_lengths) is not tuple or not query_lengths or
            any(type(n) is not int or n < 1 for n in query_lengths) or
            input_ids.ndim != 1 or target_ids.shape != input_ids.shape or
            input_ids.numel() != sum(query_lengths) or remaining.shape != (len(query_lengths),) or
            any(t.dtype not in (torch.int32, torch.int64) for t in (input_ids, target_ids, remaining)) or
            input_ids.device != target_ids.device or input_ids.device != remaining.device or
            type(eos_ids) is not tuple or any(type(i) is not int or i < 0 for i in eos_ids)):
        raise ValueError('Invalid packed greedy verification geometry/types')
    outputs, counts, matched, validity = [], [], [], []
    maximum = max(query_lengths)
    offset = 0
    for request, length in enumerate(query_lengths):
        targets = target_ids[offset:offset + length]
        drafts = input_ids[offset + 1:offset + length]
        columns = torch.arange(length, dtype=torch.int32, device=input_ids.device)
        accepted = (drafts == targets[:-1]).to(torch.int32).cumprod(0, dtype=torch.int32).sum(dtype=torch.int32)
        choices = torch.cat((drafts, targets[-1:]))
        choices = torch.where(columns == accepted, targets, choices)
        valid = remaining[request] >= 0
        advance = torch.where(valid, torch.minimum(accepted + 1, remaining[request]), 0).to(torch.int32)
        is_eos = torch.zeros_like(choices, dtype=torch.bool)
        for token in eos_ids:
            is_eos |= choices == token
        first_eos = torch.where(is_eos & (columns < advance), columns + 1, length + 1).amin()
        advance = torch.minimum(advance, first_eos).to(torch.int32)
        row = torch.where(columns < advance, choices, -1)
        outputs.append(F.pad(row, (0, maximum-length), value=-1))
        counts.append(advance)
        matched.append(accepted)
        validity.append(valid)
        offset += length
    count = torch.stack(counts)
    return GreedyVerification(torch.stack(outputs), count, count.clone(), torch.stack(matched), torch.stack(validity))


def verify_logits(logits, input_ids, query_lengths, remaining, *, eos_ids=()):
    if logits.ndim != 2 or logits.shape[0] != input_ids.numel() or logits.shape[1] < 1:
        raise ValueError('Every valid target query must retain one full-vocabulary logit row')
    # FP32 target argmax, normal deterministic greedy semantics. Random
    # sampling/rejection probabilities require a different verifier contract.
    return verify_ids(input_ids, logits.argmax(-1).to(torch.int32), query_lengths, remaining, eos_ids=eos_ids)
