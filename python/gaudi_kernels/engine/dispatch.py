"""CPU metadata dispatch. Device counts are never read back to choose a path.

These entries describe the existing measured branches, not a newly qualified
universal kernel. Actual Synapse producers are migrated separately.
"""
from dataclasses import dataclass
from .config import EngineConfig


@dataclass(frozen=True)
class Request:
    op: str
    rows: int
    n: int = 0
    k: int = 0
    device: str = 'gaudi2'

    def __post_init__(self):
        if self.device != 'gaudi2':
            raise ValueError('This qualified dispatch table supports Gaudi2 only')
        if type(self.rows) is not int or self.rows < 1:
            raise ValueError('rows must be a positive static dimension')
        if any(type(v) is not int or v < 0 for v in (self.n, self.k)):
            raise ValueError('n/k must be nonnegative integers')


@dataclass(frozen=True)
class Implementation:
    name: str
    activation: str = ''
    accumulator: str = 'fp32'
    scope: str = 'existing source branch; full-model admission is separate'


class DispatchTable:
    def __init__(self, config: EngineConfig):
        config.validate()
        self.config = config

    def resolve(self, request: Request):
        d = self.config.decode
        if request.op == 'qkv':
            if not request.n or not request.k:
                raise ValueError('QKV dispatch requires actual N/K')
            if (request.n, request.k) == (3392, 6144) and request.rows <= d.qkv.a8_max_rows:
                return Implementation('block_fp8.native_lanes.fp8_mme', 'block128_fp8')
            return Implementation('block_fp8.native_lanes.bf16_mme', 'bf16')
        if request.op == 'moe_dispatch':
            name = ('grouped' if request.rows in d.moe.dispatch.grouped_rows else
                    'compact' if request.rows in d.moe.dispatch.compact_rows else 'broadcast')
            return Implementation('mxfp4.' + name, 'bf16')
        if request.op == 'moe_gate_up':
            return Implementation('mxfp4.' + d.moe.gate_up.impl, 'bf16')
        if request.op == 'moe_down':
            return Implementation('mxfp4.' + d.moe.down.impl, 'bf16')
        if request.op == 'norm':
            return Implementation('norm.' + d.norm.impl, 'bf16')
        if request.op == 'attention_swa':
            return Implementation('attention.' + d.attention.swa, 'bf16')
        if request.op == 'attention_full':
            return Implementation('attention.' + d.attention.full, 'bf16')
        if request.op == 'tp_reduce':
            return Implementation('collective.' + d.tp_reduce.collective)
        raise ValueError(f'Unknown operation: {request.op}')

    def runtime_admission(self):
        if self.config.runtime.executor == 'pytorch':
            if self.config.runtime.flights != 1:
                raise ValueError('The serving adapter does not implement a framework flight pool')
            return 'pytorch'
        if self.config.runtime.executor == 'native_graph':
            raise ValueError('Full-model explicit producer is not qualified; refusing recorder relabeling')
        r = self.config.runtime
        if r.batch_replay:
            if any(b > 8 for b in r.buckets.batch):
                raise ValueError('Named batch replay is bounded to B<=8; model admission remains separate')
        elif r.buckets.batch != (1,) or r.buckets.tokens != (1,):
            raise ValueError('70-g native executor is qualified only for B1/T1; use the measured reference path')
        if self.config.runtime.flights != 1:
            raise ValueError('Legacy recorder does not implement the owned runtime flight pool')
        return 'legacy_native'
