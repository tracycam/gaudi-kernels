"""Paged visible KV and non-aliasing speculative write/commit descriptors.

These are allocator-supplied CPU metadata, not a KV allocator or a device
transaction. Applying selected prefixes requires completion and backend-specific
page promotion/tail copy. A lease is retained until reads/writes finish.
"""
from dataclasses import dataclass
from .token_batch import TokenBatch, _integer


@dataclass(frozen=True)
class KVPageView:
    lease_id: str
    request_ids: tuple[str, ...]
    committed_lengths: tuple[int, ...]
    page_tables: tuple[tuple[int, ...], ...]
    block_size: int
    allocated_pages: int
    sliding_window: int = 0

    def __post_init__(self):
        if type(self.lease_id) is not str or not self.lease_id:
            raise ValueError('KV storage requires an allocator lease identity')
        if (type(self.request_ids) is not tuple or not self.request_ids or
                any(type(rid) is not str or not rid for rid in self.request_ids) or
                len(set(self.request_ids)) != len(self.request_ids)):
            raise ValueError('Invalid KV request ownership')
        if (type(self.committed_lengths) is not tuple or type(self.page_tables) is not tuple or
                len(self.committed_lengths) != len(self.request_ids) or
                len(self.page_tables) != len(self.request_ids)):
            raise ValueError('KV lengths/pages must match requests')
        _integer(self.block_size, 'block_size', minimum=1)
        _integer(self.allocated_pages, 'allocated_pages', minimum=1)
        _integer(self.sliding_window, 'sliding_window')
        _integer(self.block_size * self.allocated_pages, 'KV slot capacity', minimum=1)
        for i, (length, pages) in enumerate(zip(self.committed_lengths, self.page_tables)):
            _integer(length, 'committed_length')
            start = max(0, length - self.sliding_window) if self.sliding_window else 0
            count = (length + self.block_size - 1) // self.block_size - start // self.block_size
            if type(pages) is not tuple or len(pages) != count:
                raise ValueError('Visible logical page coverage differs from committed KV')
            if any(type(page) is not int or not 0 <= page < self.allocated_pages for page in pages):
                raise ValueError('Invalid physical KV page')
            intervals = sorted(self.visible_ranges(i))
            if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                raise ValueError('One request aliases two visible KV positions')

    def visible_ranges(self, index):
        """Physical [start,end) ranges, without per-query copied page tables."""
        end = self.committed_lengths[index]
        start = max(0, end - self.sliding_window) if self.sliding_window else 0
        first_page = start // self.block_size
        ranges = []
        for offset, physical_page in enumerate(self.page_tables[index]):
            logical_start = (first_page + offset) * self.block_size
            lo = max(start, logical_start) - logical_start
            hi = min(end, logical_start + self.block_size) - logical_start
            if hi > lo:
                ranges.append((physical_page * self.block_size + lo,
                               physical_page * self.block_size + hi))
        return tuple(ranges)


@dataclass(frozen=True)
class RequestKVCommit:
    request_id: str
    base_length: int
    new_length: int
    promotions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class KVCommit:
    lease_id: str
    requests: tuple[RequestKVCommit, ...]
    discarded_slots: tuple[int, ...]


@dataclass(frozen=True)
class KVStagingPlan:
    base: KVPageView
    batch: TokenBatch
    staging_slots: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if type(self.base) is not KVPageView or type(self.batch) is not TokenBatch:
            raise ValueError('Staging requires an explicit KV view and token batch')
        if self.base.request_ids != self.batch.request_ids:
            raise ValueError('KV and token request ownership/order differ')
        if (type(self.staging_slots) is not tuple or
                len(self.staging_slots) != len(self.batch.requests)):
            raise ValueError('Staging slots must match requests')
        protected = tuple(interval for i in range(len(self.base.request_ids))
                          for interval in self.base.visible_ranges(i))
        used = set()
        limit = self.base.block_size * self.base.allocated_pages
        for request, length, slots in zip(self.batch.requests, self.base.committed_lengths, self.staging_slots):
            if request.start_position != length:
                raise ValueError('Query start differs from committed KV cursor')
            if type(slots) is not tuple or len(slots) != request.query_length:
                raise ValueError('Every valid query needs one declared candidate KV slot')
            for slot in slots:
                if type(slot) is not int or not 0 <= slot < limit:
                    raise ValueError('Candidate KV slot exceeds storage')
                if slot in used or any(lo <= slot < hi for lo, hi in protected):
                    raise ValueError('Candidate writes alias live KV or another query')
                used.add(slot)

    def select_commit(self, prefix_lengths, *, current_view):
        """Select input-query prefixes; emitted-token counts are not inferred.

        Pure descriptor construction. Device completion, promotion, event
        ordering and allocator updates are a separate backend responsibility.
        """
        if current_view != self.base:
            raise ValueError('Stale KV lease or committed state')
        if type(prefix_lengths) is not tuple or len(prefix_lengths) != len(self.batch.requests):
            raise ValueError('Commit prefixes must match every request')
        selected, discarded = [], []
        for request, base, slots, count in zip(self.batch.requests, self.base.committed_lengths,
                                               self.staging_slots, prefix_lengths):
            if type(count) is not int or not 0 <= count <= request.query_length:
                raise ValueError('Commit prefix exceeds computed query capacity')
            promotions = tuple((base + i, slot) for i, slot in enumerate(slots[:count]))
            selected.append(RequestKVCommit(request.request_id, base, base + count, promotions))
            discarded.extend(slots[count:])
        return KVCommit(self.base.lease_id, tuple(selected), tuple(discarded))
