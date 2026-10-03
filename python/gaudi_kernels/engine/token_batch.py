"""Packed token/query and result contracts; no Torch, HPU, or model constants.

Lengths describe valid rows. Capacity padding exists only at the end of the
packed arrays, never as max(query_length) rows for every request.
"""
from dataclasses import dataclass
from itertools import accumulate

I32_MAX = 2**31 - 1


def _integer(value, name, *, minimum=0):
    if type(value) is not int or not minimum <= value <= I32_MAX:
        raise ValueError(f'{name} must be an integer in [{minimum}, {I32_MAX}]')


@dataclass(frozen=True)
class RequestTokens:
    request_id: str
    token_ids: tuple[int, ...]
    start_position: int
    kind: str
    output_rows: tuple[int, ...] | None = None

    def __post_init__(self):
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError('A nonempty request ID is required')
        if type(self.token_ids) is not tuple or not self.token_ids:
            raise ValueError('token_ids must be a nonempty tuple')
        _integer(self.start_position, 'start_position')
        for token in self.token_ids:
            _integer(token, 'token_id')
        if self.start_position + len(self.token_ids) > I32_MAX:
            raise ValueError('Query positions exceed the index ABI')
        if self.kind not in ('decode', 'prefill', 'verify'):
            raise ValueError('Unknown query kind')
        if self.kind == 'decode' and len(self.token_ids) != 1:
            raise ValueError('Ordinary decode has exactly one scheduled input')
        selected = self.output_rows
        if selected is None:
            selected = (tuple(range(len(self.token_ids))) if self.kind == 'verify'
                        else (0,) if self.kind == 'decode' else ())
            object.__setattr__(self, 'output_rows', selected)
        if (type(selected) is not tuple or
                any(type(row) is not int or not 0 <= row < len(self.token_ids) for row in selected) or
                tuple(sorted(set(selected))) != selected):
            raise ValueError('Output rows must be distinct, ordered local query indices')

    @property
    def query_length(self):
        return len(self.token_ids)


@dataclass(frozen=True)
class BatchCapacity:
    token_rows: int
    requests: int
    logits_rows: int

    def __post_init__(self):
        _integer(self.token_rows, 'token_rows', minimum=1)
        _integer(self.requests, 'requests', minimum=1)
        _integer(self.logits_rows, 'logits_rows')


@dataclass(frozen=True)
class TokenBatch:
    requests: tuple[RequestTokens, ...]
    capacity: BatchCapacity | None = None

    def __post_init__(self):
        self._validate_requests(RequestTokens)

    def _validate_requests(self, request_type):
        if (type(self.requests) is not tuple or not self.requests or
                any(type(request) is not request_type for request in self.requests)):
            raise ValueError('A batch requires a tuple of declared requests')
        if len(set(self.request_ids)) != len(self.requests):
            raise ValueError('Duplicate request ID')
        if self.capacity is None:
            object.__setattr__(self, 'capacity', BatchCapacity(
                self.num_tokens, len(self.requests), len(self.logits_indices)))
        if type(self.capacity) is not BatchCapacity:
            raise ValueError('Invalid batch capacity')
        if (self.num_tokens > self.capacity.token_rows or
                len(self.requests) > self.capacity.requests or
                len(self.logits_indices) > self.capacity.logits_rows):
            raise ValueError('Valid work exceeds the declared batch capacity')

    @property
    def request_ids(self):
        return tuple(request.request_id for request in self.requests)

    @property
    def query_lengths(self):
        return tuple(request.query_length for request in self.requests)

    @property
    def num_tokens(self):
        return sum(self.query_lengths)

    @property
    def query_start_loc(self):
        return (0, *accumulate(self.query_lengths))

    @property
    def logits_indices(self):
        return tuple(start + row for start, request in zip(self.query_start_loc, self.requests)
                     for row in request.output_rows)

    @property
    def logits_owners(self):
        return tuple((request.request_id, row) for request in self.requests for row in request.output_rows)

    @property
    def capacity_key(self):
        """Only a layout capacity, not a complete kernel/graph compatibility key."""
        return (self.capacity.token_rows, self.capacity.requests, self.capacity.logits_rows)

    def encoded(self):
        """CPU values for bounded upload; padded request offsets stay monotonic."""
        c = self.capacity
        token_ids = tuple(token for request in self.requests for token in request.token_ids)
        positions = tuple(position for request in self.requests
                          for position in range(request.start_position,
                                                request.start_position + request.query_length))
        row_requests = tuple(i for i, request in enumerate(self.requests) for _ in request.token_ids)
        token_pad = c.token_rows - self.num_tokens
        request_pad = c.requests - len(self.requests)
        logits_pad = c.logits_rows - len(self.logits_indices)
        return {
            'token_ids': token_ids + (0,) * token_pad,
            'positions': positions + (0,) * token_pad,
            'row_requests': row_requests + (-1,) * token_pad,
            'query_start_loc': self.query_start_loc + (self.num_tokens,) * request_pad,
            'query_lengths': self.query_lengths + (0,) * request_pad,
            'seq_lens': tuple(request.start_position + request.query_length for request in self.requests)
                        + (0,) * request_pad,
            'logits_indices': self.logits_indices + (-1,) * logits_pad,
        }

    def validate_result(self, result):
        if type(result) is not StepResult or tuple(row.request_id for row in result.requests) != self.request_ids:
            raise ValueError('Result request ownership/order differs from input')
        for request, row in zip(self.requests, result.requests):
            if row.computed_queries != request.query_length:
                raise ValueError('Incomplete computed-query coverage')
            if len(row.emitted_tokens) > len(request.output_rows):
                raise ValueError('Result emits more tokens than the selected query capacity')
            if request.kind != 'verify' and row.committed_queries != row.computed_queries:
                raise ValueError('Non-speculative work must commit all computed input queries')
        return result

    def query_schedule(self):
        """Keep spans/ownership while token values travel through device buffers."""
        return QueryBatch(tuple(RequestQueries(r.request_id, r.query_length, r.start_position,
                                               r.kind, r.output_rows) for r in self.requests), self.capacity)


@dataclass(frozen=True)
class RequestQueries:
    request_id: str
    query_length: int
    start_position: int
    kind: str
    output_rows: tuple[int, ...] | None = None

    def __post_init__(self):
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError('A nonempty request ID is required')
        _integer(self.query_length, 'query_length', minimum=1)
        _integer(self.start_position, 'start_position')
        if self.start_position + self.query_length > I32_MAX:
            raise ValueError('Query positions exceed the index ABI')
        if self.kind not in ('decode', 'prefill', 'verify') or (self.kind == 'decode' and self.query_length != 1):
            raise ValueError('Invalid query kind/extent')
        selected = self.output_rows
        if selected is None:
            selected = tuple(range(self.query_length)) if self.kind == 'verify' else (0,) if self.kind == 'decode' else ()
            object.__setattr__(self, 'output_rows', selected)
        if (type(selected) is not tuple or
                any(type(row) is not int or not 0 <= row < self.query_length for row in selected) or
                tuple(sorted(set(selected))) != selected):
            raise ValueError('Output rows must be distinct, ordered local query indices')


@dataclass(frozen=True)
class QueryBatch(TokenBatch):
    """Same packed spans, with device-owned token values supplied separately.

    No dummy CPU IDs: encoding token values is deliberately unavailable.
    A target must receive validated device bindings before execution.
    """
    requests: tuple[RequestQueries, ...]

    def __post_init__(self):
        self._validate_requests(RequestQueries)

    def encoded(self):
        raise ValueError('Device query schedule has no CPU token values to encode')

    def query_schedule(self):
        return self


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    computed_queries: int
    committed_queries: int
    emitted_tokens: tuple[int, ...]

    def __post_init__(self):
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError('A nonempty result request ID is required')
        _integer(self.computed_queries, 'computed_queries')
        _integer(self.committed_queries, 'committed_queries')
        if self.committed_queries > self.computed_queries:
            raise ValueError('Committed queries exceed computed input')
        if type(self.emitted_tokens) is not tuple:
            raise ValueError('Emitted tokens must be a tuple')
        for token in self.emitted_tokens:
            _integer(token, 'emitted_token')


@dataclass(frozen=True)
class StepResult:
    requests: tuple[RequestResult, ...]

    def __post_init__(self):
        if (type(self.requests) is not tuple or any(type(row) is not RequestResult for row in self.requests) or
                len(set(row.request_id for row in self.requests)) != len(self.requests)):
            raise ValueError('Invalid result request set')
