"""Bounded, paged packed attention for the actual target model.

This FP32 correctness path tiles scores and keeps GQA KV heads shared. It is
not yet the fused/captured performance path. Cache storage is exclusively owned
by a session, never borrowed from active vLLM requests.
"""
from dataclasses import dataclass
import math

import torch

from gaudi_kernels.engine.token_batch import TokenBatch


@dataclass
class PackedAttentionMetadata:
    batch: TokenBatch
    session: object
    candidate_slots: tuple
    slot_mapping: torch.Tensor
    visible_slots: tuple
    key_positions: tuple
    query_positions: tuple
    is_prompt: bool = True


class PackedKVSession:
    """Exclusive test/target arena with prefix promotion, not a vLLM allocator.

    Arbitrary token slots allow rejection without overwriting a live page tail.
    Production page ownership/COW and device completion tickets remain separate.
    A caller must complete device work before commit/abort or arena reuse.
    """
    def __init__(self, geometries, capacity, *, device, dtype=torch.bfloat16):
        if type(capacity) is not int or capacity < 1 or not geometries:
            raise ValueError('Nonempty cache geometry and positive capacity required')
        self.device = torch.device(device)
        self.dtype = dtype
        self.capacity = capacity
        self.caches = {}
        for name, (heads, key_dim, value_dim) in geometries.items():
            if any(type(v) is not int or v < 1 for v in (heads, key_dim, value_dim)):
                raise ValueError('Invalid KV geometry')
            self.caches[name] = (torch.zeros((capacity, heads, key_dim), device=device, dtype=dtype),
                                 torch.zeros((capacity, heads, value_dim), device=device, dtype=dtype))
        self.committed = {}
        self.pending = None

    def prepare(self, batch):
        if type(batch) is not TokenBatch or self.pending is not None:
            raise ValueError('Explicit batch required with no pending transaction')
        for request in batch.requests:
            if request.start_position != len(self.committed.get(request.request_id, ())):
                raise ValueError('Request cursor differs from committed KV')
        occupied = {slot for slots in self.committed.values() for slot in slots}
        free = [slot for slot in range(self.capacity) if slot not in occupied]
        if batch.num_tokens > len(free):
            raise ValueError('Insufficient candidate KV capacity')
        candidates, visible, key_positions, query_positions = [], [], [], []
        for request, offset in zip(batch.requests, batch.query_start_loc):
            slots = tuple(free[offset:offset + request.query_length])
            candidates.append(slots)
            history = self.committed.get(request.request_id, ())
            visible.append(torch.tensor((*history, *slots), device=self.device, dtype=torch.int64))
            key_positions.append(torch.arange(request.start_position + request.query_length, device=self.device))
            query_positions.append(torch.arange(request.start_position,
                                                request.start_position + request.query_length, device=self.device))
        self.pending = PackedAttentionMetadata(batch, self, tuple(candidates),
            torch.tensor(free[:batch.num_tokens], device=self.device, dtype=torch.int64),
            tuple(visible), tuple(key_positions), tuple(query_positions))
        return self.pending

    def commit(self, metadata, prefix_lengths):
        if metadata is not self.pending or len(prefix_lengths) != len(metadata.batch.requests):
            raise ValueError('Stale or incomplete KV transaction')
        # Validate every request before changing any state.
        if any(type(n) is not int or not 0 <= n <= request.query_length
               for n, request in zip(prefix_lengths, metadata.batch.requests)):
            raise ValueError('Commit must be a computed input prefix')
        for request, slots, count in zip(metadata.batch.requests, metadata.candidate_slots, prefix_lengths):
            history = self.committed.get(request.request_id, ())
            self.committed[request.request_id] = (*history, *slots[:count])
        self.pending = None

    def abort(self, metadata):
        if metadata is not self.pending:
            raise ValueError('Stale KV transaction')
        self.pending = None


def tiled_attention(query, keys, values, query_positions, key_positions, *, scale,
                    sliding_window=None, sinks=None, tile_size=256, slot_indices=None, key_valid=None):
    """Online FP32 softmax; workspace O(Hq * R_request * tile_size)."""
    rows, heads, key_dim = query.shape
    kv_heads = keys.shape[1]
    if (heads % kv_heads or keys.shape[2] != key_dim or values.shape[:2] != keys.shape[:2]
            or type(tile_size) is not int or tile_size < 1 or not math.isfinite(scale)):
        raise ValueError('Invalid packed attention geometry')
    groups = heads // kv_heads
    q = query.float().transpose(0, 1).reshape(kv_heads, groups, rows, key_dim)
    shape = (kv_heads, groups, rows, 1)
    maximum = (torch.full(shape, float('-inf'), device=query.device) if sinks is None else
               sinks.float().reshape(kv_heads, groups, 1, 1).expand(shape))
    denominator = torch.zeros(shape, device=query.device) if sinks is None else torch.ones(shape, device=query.device)
    numerator = torch.zeros((*shape[:-1], values.shape[2]), device=query.device)
    length = keys.shape[0] if slot_indices is None else slot_indices.numel()
    for offset in range(0, length, tile_size):
        end = min(offset + tile_size, length)
        if slot_indices is None:
            key_tile, value_tile = keys[offset:end], values[offset:end]
        else:
            slots = slot_indices[offset:end]
            key_tile, value_tile = keys.index_select(0, slots), values.index_select(0, slots)
        if key_valid is not None:
            valid = key_valid[offset:end]
            # Even a zero softmax weight times a NaN value yields NaN. Sanitize
            # invalid padding before either matmul, not just the score mask.
            key_tile = torch.where(valid[:, None, None], key_tile, torch.zeros_like(key_tile))
            value_tile = torch.where(valid[:, None, None], value_tile, torch.zeros_like(value_tile))
        k = key_tile.float().permute(1, 2, 0).unsqueeze(1)
        v = value_tile.float().transpose(0, 1).unsqueeze(1)
        scores = torch.matmul(q, k) * scale
        positions = key_positions[offset:end]
        allowed = positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if key_valid is not None:
            allowed &= key_valid[offset:end].unsqueeze(0)
        if sliding_window:
            allowed &= positions.unsqueeze(0) > query_positions.unsqueeze(1) - sliding_window
        scores = scores.masked_fill(~allowed, float('-inf'))
        next_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        # Whole tiles can be masked by SWA; -inf - -inf must not poison sums.
        safe_maximum = torch.where(torch.isfinite(next_maximum), next_maximum, torch.zeros_like(next_maximum))
        rescale = torch.exp(maximum - safe_maximum)
        probabilities = torch.exp(scores - safe_maximum)
        numerator = numerator * rescale + torch.matmul(probabilities, v)
        denominator = denominator * rescale + probabilities.sum(-1, keepdim=True)
        maximum = next_maximum
    result = numerator / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return result.reshape(heads, rows, values.shape[2]).transpose(0, 1).to(query.dtype)


def forward_packed(impl, layer, query, key, value, metadata, output=None):
    session = metadata.session
    if session.pending is not metadata or layer.layer_name not in session.caches:
        raise ValueError('Unowned or stale packed attention cache')
    if impl.kv_sharing_target_layer_name is not None or impl.alibi_slopes is not None:
        raise ValueError('Packed target path does not yet admit KV sharing or ALiBi')
    rows = metadata.batch.num_tokens
    if key is None or value is None or query.shape[0] != rows:
        raise ValueError('Packed target requires exactly the valid query rows')
    key_cache, value_cache = session.caches[layer.layer_name]
    query = query.reshape(rows, impl.num_heads, impl.head_size)
    key = key.reshape(rows, impl.num_kv_heads, impl.head_size)
    value = value.reshape(rows, impl.num_kv_heads, impl.head_size_v)
    key_cache = key_cache.index_copy_(0, metadata.slot_mapping, key.to(key_cache.dtype))
    value_cache = value_cache.index_copy_(0, metadata.slot_mapping, value.to(value_cache.dtype))
    session.caches[layer.layer_name] = (key_cache, value_cache)
    segments = []
    for index, request in enumerate(metadata.batch.requests):
        begin, end = metadata.batch.query_start_loc[index:index + 2]
        key_valid = None
        if hasattr(metadata, 'history_capacity'):
            length = metadata.visible_lengths[index]
            if impl.sliding_window:
                width = min(metadata.history_capacity, impl.sliding_window + request.query_length - 1)
                base = (metadata.query_positions[index][0] - impl.sliding_window + 1).clamp(min=0)
                positions = torch.arange(width, device=query.device) + base
                indices = positions.clamp(max=metadata.history_capacity - 1)
                slots = metadata.visible_slots[index].index_select(0, indices)
            else:
                positions = metadata.key_positions[index]
                slots = metadata.visible_slots[index]
            key_valid = positions < length
        else:
            past_start = max(0, request.start_position - impl.sliding_window + 1) if impl.sliding_window else 0
            positions = metadata.key_positions[index][past_start:]
            slots = metadata.visible_slots[index][past_start:]
        segments.append(tiled_attention(query[begin:end], key_cache, value_cache,
            metadata.query_positions[index], positions,
            scale=impl.scale, sliding_window=impl.sliding_window, sinks=impl.sinks,
            slot_indices=slots, key_valid=key_valid))
    result = torch.cat(segments)
    if output is not None:
        output.copy_(result.reshape(output.shape))
        return output
    return result.reshape(rows, -1)
