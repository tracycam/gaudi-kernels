"""Device-only commit for greedy chain verification; fixed-capacity outputs.

The query rows are [anchor, draft_0, ..., draft_(L-1)]. Before this forward,
base_kv_length excludes the anchor. Accepted draft rows plus the anchor become
visible. Speculative KV must use isolated staging: changing a length cannot
restore committed SWA entries overwritten by a ring-buffer alias.
No page allocation, host readback, stochastic acceptance or tree verification.
"""
from typing import NamedTuple
import torch

class VerifyResult(NamedTuple):
    tokens: torch.Tensor
    token_count: torch.Tensor
    accepted_drafts: torch.Tensor
    next_token: torch.Tensor
    committed_kv_length: torch.Tensor

def verify_greedy(predicted_next, drafts, base_kv_length, draft_lengths, active, *, remaining, eos_id):
    """All arguments are device tensors, including lengths and active masks.

    predicted_next: int [B,L+1], argmax after each input query row.
    drafts: int [B,L], including fixed-capacity padding.
    base_kv_length, draft_lengths, active, remaining: [B]. remaining is the
    output budget, not a draft-length limit. EOS is included in emitted tokens.
    """
    if drafts.ndim != 2 or predicted_next.shape != (drafts.shape[0], drafts.shape[1] + 1):
        raise ValueError('Expected drafts[B,L] and predicted_next[B,L+1]')
    (b, l) = drafts.shape
    if any((t.shape != (b,) for t in (base_kv_length, draft_lengths, active, remaining))):
        raise ValueError('Length/state tensors must have shape [B]')
    columns = torch.arange(l, device=drafts.device, dtype=torch.int32).unsqueeze(0)
    match = (predicted_next[:, :l] == drafts) & (columns < draft_lengths.unsqueeze(1))
    accepted = torch.cumprod(match.to(torch.int32), dim=1, dtype=torch.int32).sum(dim=1, dtype=torch.int32)
    accepted = torch.where(active, accepted, torch.zeros_like(accepted))
    correction = predicted_next.gather(1, accepted.to(torch.int64).unsqueeze(1)).squeeze(1)
    columns = torch.arange(l + 1, device=drafts.device, dtype=torch.int32).unsqueeze(0)
    tokens = torch.cat((drafts, torch.zeros_like(correction).unsqueeze(1)), dim=1)
    tokens = torch.where(columns == accepted.unsqueeze(1), correction.unsqueeze(1), tokens)
    count = torch.where(active, torch.minimum(accepted + 1, remaining.clamp_min(0)), torch.zeros_like(accepted))
    eos_end = torch.where((tokens == eos_id) & (columns < count.unsqueeze(1)),
                          columns + 1, l + 2).amin(dim=1)
    count = torch.minimum(count, eos_end).to(torch.int32)
    # After budget/EOS truncation, the next anchor is the last emitted token.
    next_token = tokens.gather(1, (count-1).clamp_min(0).to(torch.int64).unsqueeze(1)).squeeze(1)
    next_token = torch.where(count > 0, next_token, torch.zeros_like(next_token))
    tokens = torch.where(columns < count.unsqueeze(1), tokens, torch.zeros_like(tokens))
    committed = base_kv_length + count
    emitted_drafts = torch.minimum(accepted, count)
    return VerifyResult(tokens, count, emitted_drafts, next_token, committed)

def commit_rows(base_kv_length, query_lengths, active):
    """Prefill/decode commit uses the same device length state as verification."""
    return torch.where(active, base_kv_length + query_lengths, base_kv_length)
