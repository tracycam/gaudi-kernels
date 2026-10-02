"""Tensor-only speculative state primitives, awaiting full executor integration.

KV staging is disjoint from the committed SWA ring. Capture may unroll K, but
no accepted length or token passes through Python during replay. Payload uses
[B, W, ...] here; production paged KV needs its own binding/layout adapter.
"""
import torch
import torch.nn.functional as F


def visibility(committed_positions, lengths, capacity, window):
    """Target causal/SWA mask over [committed ring, isolated verify staging]."""
    b, w = committed_positions.shape
    assert window <= w and capacity <= w
    columns = torch.arange(capacity, device=lengths.device, dtype=lengths.dtype)
    queries = lengths[:, None] + columns
    keys = torch.cat((committed_positions, queries), dim=1)
    mask = (keys[:, None, :] >= 0) & (keys[:, None, :] <= queries[:, :, None])
    return mask & (keys[:, None, :] > queries[:, :, None] - window)


def commit_ring(ring, committed_positions, lengths, staging, advance):
    """Commit anchor+accepted rows; discarded rows cannot overwrite live KV.

    Caller validates static shapes and device advance in [0,T] via the accept
    contract. Each T<=W destination is unique; invalid writes copy old values.
    Returns tensors, leaving input ownership to the surrounding graph plan.
    """
    b, w = ring.shape[:2]
    t = staging.shape[1]
    assert t <= w and staging.shape[0] == b and staging.shape[2:] == ring.shape[2:]
    assert staging.dtype == ring.dtype
    columns = torch.arange(t, device=lengths.device, dtype=lengths.dtype)
    positions = lengths[:, None] + columns
    slots = (positions % w).long()
    valid = columns < advance[:, None]
    shape = (b, t) + (1,) * (ring.ndim-2)
    indices = slots.reshape(shape).expand_as(staging)
    previous = torch.gather(ring, 1, indices)
    values = torch.where(valid.reshape(shape), staging, previous)
    updated = ring.scatter(1, indices, values)
    previous_positions = torch.gather(committed_positions, 1, slots)
    tags = committed_positions.scatter(1, slots, torch.where(valid, positions, previous_positions))
    return updated, tags, lengths + advance


def dspark_greedy(base_logits, anchor, markov_w1, markov_w2, draft_to_target):
    """Dense unquantized DSpark Markov head, with fixed K captured on device.

    Checkpoint-specific backbone/position layout remains a separate adapter.
    w1 indexes TARGET ids. w2 emits DRAFT vocabulary logits. The map contains
    absolute target ids (not vLLM's stored offsets). No top-k truncation or
    fabricated Markov weights is allowed. This is greedy only.
    """
    b, k, vocab = base_logits.shape
    assert anchor.shape == (b,) and draft_to_target.shape == (vocab,)
    assert markov_w2.shape == (vocab, markov_w1.shape[1])
    previous = anchor
    output = []
    for i in range(k):
        bias = F.linear(F.embedding(previous.long(), markov_w1), markov_w2)
        logits = base_logits[:, i].float() + bias.float()
        chosen = logits.argmax(-1)
        previous = draft_to_target[chosen]
        output.append(previous)
    return torch.stack(output, 1)
