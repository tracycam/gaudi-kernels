"""Small packed target-layer semantic probe, not a production attention backend.

Shared projections execute once across valid rows. Per-request FP32 attention
is a diagnostic implementation with materialized scores/KV concatenation.
It does not establish physical traffic, native replay, or full-model admission.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

from gaudi_kernels.engine.token_batch import TokenBatch


@dataclass(frozen=True)
class LayerSpec:
    hidden_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    sliding_window: int = 0
    epsilon: float = 1e-6

    def __post_init__(self):
        if (any(type(v) is not int or v < 1 for v in
                (self.hidden_size, self.query_heads, self.kv_heads, self.head_dim)) or
                self.hidden_size != self.query_heads * self.head_dim or
                self.query_heads % self.kv_heads or
                type(self.sliding_window) is not int or self.sliding_window < 0 or
                not math.isfinite(self.epsilon) or self.epsilon <= 0):
            raise ValueError('Unsupported reference layer geometry')


@dataclass
class TargetOutput:
    hidden: torch.Tensor
    candidate_keys: tuple
    candidate_values: tuple
    projection_calls: dict


def _norm(x, weight, epsilon):
    fp32 = x.float()
    return (fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + epsilon)
            * weight.float()).to(x.dtype)


def _split(projection, spec):
    q_size = spec.query_heads * spec.head_dim
    kv_size = spec.kv_heads * spec.head_dim
    q, k, v = projection.split((q_size, kv_size, kv_size), dim=-1)
    return (q.reshape(-1, spec.query_heads, spec.head_dim),
            k.reshape(-1, spec.kv_heads, spec.head_dim), v.reshape(-1, spec.kv_heads, spec.head_dim))


def _attention(q, key, value, positions, key_start, spec):
    # GQA shares KV storage; expansion is only in this semantic reference.
    groups = spec.query_heads // spec.kv_heads
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    key_positions = torch.arange(key_start, key_start + key.shape[0], device=q.device)
    allowed = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
    if spec.sliding_window:
        allowed &= key_positions.unsqueeze(0) > positions.unsqueeze(1) - spec.sliding_window
    scores = torch.matmul(q.transpose(0, 1).float(), key.permute(1, 2, 0).float())
    scores = scores / math.sqrt(spec.head_dim)
    scores = scores.masked_fill(~allowed.unsqueeze(0), float('-inf'))
    probabilities = torch.softmax(scores, dim=-1)
    result = torch.matmul(probabilities, value.transpose(0, 1).float())
    return result.transpose(0, 1).reshape(q.shape[0], spec.hidden_size).to(q.dtype)


def _validate(batch, hidden, norm_weight, qkv_weight, out_weight, histories, spec):
    if type(batch) is not TokenBatch or type(spec) is not LayerSpec:
        raise ValueError('Explicit batch and layer contracts are required')
    qkv_size = (spec.query_heads + 2 * spec.kv_heads) * spec.head_dim
    if (hidden.ndim != 2 or hidden.shape != (batch.capacity.token_rows, spec.hidden_size) or
            norm_weight.shape != (spec.hidden_size,) or
            qkv_weight.shape != (qkv_size, spec.hidden_size) or
            out_weight.shape != (spec.hidden_size, spec.hidden_size) or
            len(histories) != len(batch.requests)):
        raise ValueError('Invalid reference layer inputs')
    if any(t.device != hidden.device or t.dtype != hidden.dtype
           for t in (norm_weight, qkv_weight, out_weight)):
        raise ValueError('Reference layer device/dtype mismatch')
    for request, (key, value) in zip(batch.requests, histories):
        expected = (request.start_position, spec.kv_heads, spec.head_dim)
        if (key.shape != expected or value.shape != expected or
                key.device != hidden.device or value.device != hidden.device or
                key.dtype != hidden.dtype or value.dtype != hidden.dtype):
            raise ValueError('History must match the declared request cursor and geometry')


def packed_target(batch, hidden, norm_weight, qkv_weight, out_weight, histories, spec):
    _validate(batch, hidden, norm_weight, qkv_weight, out_weight, histories, spec)
    valid = hidden[:batch.num_tokens]
    q, keys, values = _split(F.linear(_norm(valid, norm_weight, spec.epsilon), qkv_weight), spec)
    attended, candidates_k, candidates_v = [], [], []
    for request, offset, (past_k, past_v) in zip(batch.requests, batch.query_start_loc, histories):
        end = offset + request.query_length
        new_k, new_v = keys[offset:end], values[offset:end]
        key, value = torch.cat((past_k, new_k)), torch.cat((past_v, new_v))
        positions = torch.arange(request.start_position, request.start_position + request.query_length,
                                 device=hidden.device)
        attended.append(_attention(q[offset:end], key, value, positions, 0, spec))
        candidates_k.append(new_k)
        candidates_v.append(new_v)
    output = valid + F.linear(torch.cat(attended), out_weight)
    return TargetOutput(output, tuple(candidates_k), tuple(candidates_v), {'qkv': 1, 'out': 1})


def sequential_target(batch, hidden, norm_weight, qkv_weight, out_weight, histories, spec):
    """Independent token-at-a-time causal reference with no future K/V present."""
    _validate(batch, hidden, norm_weight, qkv_weight, out_weight, histories, spec)
    output, candidates_k, candidates_v = [], [], []
    for request, offset, (past_k, past_v) in zip(batch.requests, batch.query_start_loc, histories):
        collected_k, collected_v = [], []
        for j in range(request.query_length):
            x = hidden[offset + j:offset + j + 1]
            q, k, v = _split(F.linear(_norm(x, norm_weight, spec.epsilon), qkv_weight), spec)
            collected_k.append(k)
            collected_v.append(v)
            key = torch.cat((past_k, *collected_k))
            value = torch.cat((past_v, *collected_v))
            position = torch.tensor([request.start_position + j], device=hidden.device)
            attended = _attention(q, key, value, position, 0, spec)
            output.append(x + F.linear(attended, out_weight))
        candidates_k.append(torch.cat(collected_k))
        candidates_v.append(torch.cat(collected_v))
    return TargetOutput(torch.cat(output), tuple(candidates_k), tuple(candidates_v),
                        {'qkv': batch.num_tokens, 'out': batch.num_tokens})


def fixture(*, device='cpu', sliding_window=0):
    from gaudi_kernels.engine.token_batch import BatchCapacity, RequestTokens
    generator = torch.Generator().manual_seed(37)
    spec = LayerSpec(32, 4, 2, 8, sliding_window)
    batch = TokenBatch((RequestTokens('decode', (11,), 127, 'decode'),
                        RequestTokens('verify', (21, 22, 23, 24), 126, 'verify'),
                        RequestTokens('prefill', tuple(range(128)), 0, 'prefill')),
                       BatchCapacity(144, 4, 8))

    def tensor(shape, scale=1):
        return (torch.randn(shape, generator=generator) * scale).bfloat16().to(device)

    hidden = tensor((144, 32))
    norm = torch.ones(32, dtype=torch.bfloat16, device=device)
    qkv = tensor((64, 32), .1)
    out = tensor((32, 32), .1)
    histories = tuple((tensor((r.start_position, 2, 8)), tensor((r.start_position, 2, 8)))
                      for r in batch.requests)
    return batch, hidden, norm, qkv, out, histories, spec
