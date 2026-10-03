# SPDX-License-Identifier: Apache-2.0
"""Persistent target inputs and bounded attention bindings for replay probes.

Fixed request/query extents, mutable values. This exclusively owned arena is
not the vLLM allocator, and preparation still requires a completed device step.
"""
from dataclasses import dataclass

import torch

from gaudi_kernels.engine.token_batch import TokenBatch

from gaudi_kernels.serving.executor.packed_attention import PackedAttentionMetadata


class PackedInputBuffers:
    def __init__(self, batch, device):
        self.signature = self._signature(batch)
        self.ids = torch.empty(batch.num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.empty(batch.num_tokens, dtype=torch.int64, device=device)
        self.logits_indices = torch.tensor(batch.logits_indices, dtype=torch.int64, device=device)
        self._batch = None
        if type(batch) is TokenBatch:
            self.update(batch)

    @staticmethod
    def _signature(batch):
        return (tuple(r.query_length for r in batch.requests), batch.logits_indices)

    def validate(self, batch):
        if self._signature(batch) != self.signature or self._batch is not batch:
            raise ValueError('Input buffers were not updated for this exact batch')

    def update(self, batch):
        if self._signature(batch) != self.signature:
            raise ValueError('Query extents/output selection require a different binding capacity')
        encoded = batch.encoded()
        # Values change in place; graph-owned input addresses remain stable.
        self.ids.copy_(torch.tensor(encoded['token_ids'][:batch.num_tokens], dtype=self.ids.dtype))
        self.positions.copy_(torch.tensor(encoded['positions'][:batch.num_tokens], dtype=self.positions.dtype))
        self._batch = batch

    def update_device(self, batch, token_ids, positions):
        """Feed drafter output to the target without reading device IDs on CPU."""
        if (self._signature(batch) != self.signature or token_ids.shape != self.ids.shape or
                positions.shape != self.positions.shape or token_ids.device != self.ids.device or
                positions.device != self.positions.device or
                token_ids.dtype not in (torch.int32, torch.int64) or positions.dtype not in (torch.int32, torch.int64)):
            raise ValueError('Device token/position bindings differ from declared query extents')
        self.ids.copy_(token_ids)
        self.positions.copy_(positions)
        self._batch = batch


@dataclass
class BoundAttentionMetadata(PackedAttentionMetadata):
    visible_lengths: tuple = ()
    history_capacity: int = 0
    window_page_ids: torch.Tensor | None = None
    window_page_groups: torch.Tensor | None = None
    flat_query_positions: torch.Tensor | None = None


class BoundKVSession:
    """Stable attention tensor addresses with fresh, stale-checked transactions.

    Delegates candidate allocation/commit to an exclusive PackedKVSession.
    Padding reads safe slot zero and is explicitly masked on device. SWA reads
    at most window+query_length-1 slots rather than the full history capacity.
    """
    def __init__(self, arena, query_lengths, history_capacity):
        lengths = tuple(query_lengths)
        if (not lengths or any(type(n) is not int or n < 1 for n in lengths) or
                type(history_capacity) is not int or history_capacity < max(lengths)):
            raise ValueError('Positive fixed query/history capacities required')
        self.arena = arena
        self.query_lengths = lengths
        self.history_capacity = history_capacity
        self.pending = None
        self._actual = None
        device = arena.device
        self.mapping = torch.empty(sum(lengths), device=device, dtype=torch.int64)
        self.slots = tuple(torch.empty(history_capacity, device=device, dtype=torch.int64) for _ in lengths)
        self.lengths = tuple(torch.empty((), device=device, dtype=torch.int64) for _ in lengths)
        self.query_positions = tuple(torch.empty(n, device=device, dtype=torch.int64) for n in lengths)
        self.key_positions = tuple(torch.arange(history_capacity, device=device) for _ in lengths)
        self.native_swa = getattr(arena, 'native_swa', False)
        self.window_page_ids = torch.empty(2*sum(lengths), device=device, dtype=torch.int32)
        self.window_page_groups = torch.empty_like(self.window_page_ids)
        self.flat_query_positions = torch.empty(sum(lengths), device=device, dtype=torch.int32)

    @property
    def caches(self):
        return self.arena.caches

    def prepare(self, batch):
        if self.pending is not None or tuple(r.query_length for r in batch.requests) != self.query_lengths:
            raise ValueError('Unfinished transaction or changed fixed query extents')
        if any(r.start_position + r.query_length > self.history_capacity for r in batch.requests):
            raise ValueError('History exceeds captured attention capacity')
        actual = self.arena.prepare(batch)
        try:
            self.mapping.copy_(actual.slot_mapping)
            for i, request in enumerate(batch.requests):
                length = request.start_position + request.query_length
                self.slots[i].zero_()
                self.slots[i][:length].copy_(actual.visible_slots[i])
                self.lengths[i].fill_(length)
                self.query_positions[i].copy_(actual.query_positions[i])
            self.pending = BoundAttentionMetadata(batch, self, actual.candidate_slots, self.mapping,
                self.slots, self.key_positions, self.query_positions, True, self.lengths, self.history_capacity)
            if hasattr(actual, 'window_page_ids'):
                self.window_page_ids.copy_(actual.window_page_ids)
                self.window_page_groups.copy_(actual.window_page_groups)
                self.flat_query_positions.copy_(actual.flat_query_positions)
                self.pending.window_page_ids = self.window_page_ids
                self.pending.window_page_groups = self.window_page_groups
                self.pending.flat_query_positions = self.flat_query_positions
            self._actual = actual
            return self.pending
        except BaseException:
            self.arena.abort(actual)
            raise

    def commit(self, metadata, prefix_lengths):
        if metadata is not self.pending:
            raise ValueError('Stale bound KV transaction')
        self.arena.commit(self._actual, prefix_lengths)
        self.pending = self._actual = None

    def abort(self, metadata):
        if metadata is not self.pending:
            raise ValueError('Stale bound KV transaction')
        self.arena.abort(self._actual)
        self.pending = self._actual = None
