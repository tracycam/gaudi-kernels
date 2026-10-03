# SPDX-License-Identifier: Apache-2.0
"""Exclusive page-aligned diagnostic KV arena for native SWA consumers.

Each request retains its own physical pages. Speculative suffixes are staged
beyond the live cursor; rejection does not restore or overwrite an SWA ring.
This arena is deliberately bounded and is not a serving allocator replacement.
"""
from dataclasses import dataclass

import torch

from gaudi_kernels.serving.executor.packed_attention import PackedAttentionMetadata, PackedKVSession


@dataclass
class PagedAttentionMetadata(PackedAttentionMetadata):
    window_page_ids: torch.Tensor | None = None
    window_page_groups: torch.Tensor | None = None
    flat_query_positions: torch.Tensor | None = None


class PagedKVSession(PackedKVSession):
    def __init__(self, geometries, request_capacities, *, device, dtype=torch.bfloat16, native_swa=False):
        if (not request_capacities or any(type(rid) is not str or not rid or type(n) is not int or n < 1
                                          for rid, n in request_capacities.items()) or type(native_swa) is not bool):
            raise ValueError('Explicit positive request capacities/native SWA policy required')
        self.request_capacities = dict(request_capacities)
        self.bases, extent = {}, 0
        for request, capacity in request_capacities.items():
            self.bases[request] = extent
            extent += ((capacity + 127)//128)*128
        self.native_swa = native_swa
        if native_swa:
            from gaudi_kernels.serving.executor.swa_batch_runtime import prepare
            prepare()
        super().__init__(geometries, extent, device=device, dtype=dtype)

    def prepare(self, batch):
        if self.pending is not None:
            raise ValueError('Page arena has an unfinished transaction')
        for request in batch.requests:
            if (request.request_id not in self.bases or
                    request.start_position != len(self.committed.get(request.request_id, ())) or
                    request.start_position + request.query_length > self.request_capacities[request.request_id]):
                raise ValueError('Page request ownership, cursor or capacity mismatch')
        candidates, visible, key_positions, query_positions = [], [], [], []
        mapping, pages, groups, positions = [], [], [], []
        row = 0
        for request in batch.requests:
            base = self.bases[request.request_id]
            start, end = request.start_position, request.start_position + request.query_length
            slots = tuple(range(base + start, base + end))
            candidates.append(slots)
            mapping.extend(slots)
            visible.append(torch.arange(base, base + end, device=self.device, dtype=torch.int64))
            key_positions.append(torch.arange(end, device=self.device))
            query_positions.append(torch.arange(start, end, device=self.device))
            for position in range(start, end):
                logical0 = (max(0, position - 127)//128)*128
                pages.extend((base//128 + logical0//128,
                              base//128 + logical0//128 + 1 if position >= logical0 + 128 else -1))
                groups.extend((row, row))
                positions.append(position)
                row += 1
        self.pending = PagedAttentionMetadata(batch, self, tuple(candidates),
            torch.tensor(mapping, device=self.device, dtype=torch.int64), tuple(visible),
            tuple(key_positions), tuple(query_positions), True,
            torch.tensor(pages, device=self.device, dtype=torch.int32),
            torch.tensor(groups, device=self.device, dtype=torch.int32),
            torch.tensor(positions, device=self.device, dtype=torch.int32))
        return self.pending
