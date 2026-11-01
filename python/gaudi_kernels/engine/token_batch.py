"""Packed token/query and result contracts; no Torch, HPU, or model constants.

Lengths describe valid rows. Capacity padding exists only at the end of the
packed arrays, never as max(query_length) rows for every request.
"""
from dataclasses import dataclass
from itertools import accumulate
from typing import Literal

I32_MAX = 2**31 - 1


def _integer(value, name, *, minimum=0):
    if type(value) is not int or not minimum <= value <= I32_MAX:
        raise ValueError(f'{name} must be an integer in [{minimum}, {I32_MAX}]')


@dataclass(frozen=True)
class VisibilitySpec:
    mode: Literal['causal_full', 'causal_swa', 'noncausal_swa'] = 'causal_full'
    window_size: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ('causal_full', 'causal_swa', 'noncausal_swa'):
            raise ValueError('invalid visibility mode')
        if self.mode == 'causal_full' and self.window_size is not None:
            raise ValueError('causal_full does not take window_size')
        if self.mode in ('causal_swa', 'noncausal_swa') and (
                self.window_size is None or type(self.window_size) is bool or int(self.window_size) <= 0):
            raise ValueError('sliding window visibility requires positive window_size')
        if self.window_size is not None:
            object.__setattr__(self, 'window_size', int(self.window_size))

    @property
    def is_causal(self) -> bool:
        return self.mode != 'noncausal_swa'


@dataclass(frozen=True)
class OutputSpec:
    rows: tuple[int, ...] = ()
    mode: Literal['none', 'greedy', 'topk', 'logprobs', 'full_logits'] = 'greedy'
    feature_layers: tuple[int, ...] = ()
    topk: int = 1

    def __post_init__(self) -> None:
        if self.mode not in ('none', 'greedy', 'topk', 'logprobs', 'full_logits'):
            raise ValueError('invalid output mode')
        object.__setattr__(self, 'rows', tuple(int(r) for r in self.rows))
        object.__setattr__(self, 'feature_layers', tuple(int(l) for l in self.feature_layers))
        if any(r < 0 for r in self.rows):
            raise ValueError('output rows must be non-negative')
        if self.mode == 'none' and self.rows:
            raise ValueError('none output mode requires empty rows')
        if self.mode == 'topk' and int(self.topk) <= 0:
            raise ValueError('topk must be positive')


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
class StepPlanKey:
    plan_id: str
    capacity: BatchCapacity
    visibility: VisibilitySpec = VisibilitySpec()
    output_mode: Literal['none', 'greedy', 'topk', 'logprobs', 'full_logits'] = 'greedy'
    uniform_query_len: int | None = None
    feature_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.plan_id) is not str or not self.plan_id:
            raise ValueError('A nonempty plan_id is required')
        if type(self.capacity) is not BatchCapacity:
            raise ValueError('Invalid batch capacity')
        if type(self.visibility) is not VisibilitySpec:
            raise ValueError('Invalid visibility spec')
        if self.output_mode not in ('none', 'greedy', 'topk', 'logprobs', 'full_logits'):
            raise ValueError('invalid output mode')
        if self.uniform_query_len is not None:
            _integer(self.uniform_query_len, 'uniform_query_len', minimum=1)
        object.__setattr__(self, 'feature_layers', tuple(int(l) for l in self.feature_layers))


GraphKey = StepPlanKey


@dataclass(frozen=True)
class TokenBatch:
    requests: tuple[RequestTokens, ...]
    capacity: BatchCapacity | None = None
    visibility: VisibilitySpec = VisibilitySpec()
    output_spec: OutputSpec | None = None

    def __post_init__(self):
        self._validate_requests(RequestTokens)

    def _validate_requests(self, request_type):
        if (type(self.requests) is not tuple or not self.requests or
                any(type(request) is not request_type for request in self.requests)):
            raise ValueError('A batch requires a tuple of declared requests')
        if len(set(self.request_ids)) != len(self.requests):
            raise ValueError('Duplicate request ID')
        if type(self.visibility) is not VisibilitySpec:
            raise ValueError('Invalid visibility spec')
        if self.output_spec is None:
            default_rows = self.logits_keep
            object.__setattr__(
                self,
                'output_spec',
                OutputSpec(rows=default_rows, mode='greedy' if default_rows else 'none'),
            )
        if type(self.output_spec) is not OutputSpec:
            raise ValueError('Invalid output spec')
        if any(row >= self.num_tokens for row in self.output_spec.rows):
            raise ValueError('Output spec rows exceed valid batch tokens')
        required_logits = max(len(self.logits_indices), len(self.output_spec.rows))
        if self.capacity is None:
            object.__setattr__(self, 'capacity', BatchCapacity(
                self.num_tokens, len(self.requests), required_logits))
        if type(self.capacity) is not BatchCapacity:
            raise ValueError('Invalid batch capacity')
        if (self.num_tokens > self.capacity.token_rows or
                len(self.requests) > self.capacity.requests or
                required_logits > self.capacity.logits_rows):
            raise ValueError('Valid work exceeds the declared batch capacity')

    @property
    def request_ids(self):
        return tuple(request.request_id for request in self.requests)

    @property
    def query_lengths(self):
        return tuple(request.query_length for request in self.requests)

    @property
    def uniform_query_len(self) -> int | None:
        lengths = self.query_lengths
        return lengths[0] if len(set(lengths)) == 1 else None

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
    def logits_keep(self):
        return self.logits_indices

    @property
    def logits_owners(self):
        return tuple((request.request_id, row) for request in self.requests for row in request.output_rows)

    @property
    def capacity_key(self):
        """Only a layout capacity, not a complete kernel/graph compatibility key."""
        return (self.capacity.token_rows, self.capacity.requests, self.capacity.logits_rows)

    def plan_key(
        self,
        plan_id: str = 'target',
        output_mode: str | None = None,
        feature_layers: tuple[int, ...] = (),
    ) -> StepPlanKey:
        mode = self.output_spec.mode if output_mode is None else output_mode
        layers = tuple(int(l) for l in feature_layers) if feature_layers else self.output_spec.feature_layers
        return StepPlanKey(
            plan_id=plan_id,
            capacity=self.capacity,
            visibility=self.visibility,
            output_mode=mode,
            uniform_query_len=self.uniform_query_len,
            feature_layers=layers,
        )

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
        return QueryBatch(
            tuple(RequestQueries(r.request_id, r.query_length, r.start_position,
                                 r.kind, r.output_rows) for r in self.requests),
            self.capacity,
            self.visibility,
            self.output_spec,
        )


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
    capacity: BatchCapacity | None = None
    visibility: VisibilitySpec = VisibilitySpec()
    output_spec: OutputSpec | None = None

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

    @property
    def has_uncommitted_bonus(self) -> bool:
        return len(self.emitted_tokens) > self.committed_queries

    @property
    def next_anchor(self) -> int | None:
        return self.emitted_tokens[-1] if self.emitted_tokens else None

    @property
    def materialized_kv_delta(self) -> int:
        return self.committed_queries


StateCommit = RequestResult


@dataclass(frozen=True)
class StepResult:
    requests: tuple[RequestResult, ...]

    def __post_init__(self):
        if (type(self.requests) is not tuple or any(type(row) is not RequestResult for row in self.requests) or
                len(set(row.request_id for row in self.requests)) != len(self.requests)):
            raise ValueError('Invalid result request set')

