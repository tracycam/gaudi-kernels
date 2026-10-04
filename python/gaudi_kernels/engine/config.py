"""Validated, immutable configuration. No environment-variable selection.

JSON is used so the bootstrap does not depend on PyYAML or import Torch. All
keys and values are checked recursively; coercions such as true -> 1 or
"false" -> True are forbidden. Arithmetic choices and diagnostics are separate.
"""
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Parallel:
    tp: Literal[8] = 8
    ep: Literal[1] = 1


@dataclass(frozen=True)
class QKV:
    impl: Literal['block_fp8_mixed', 'block_fp8_a16'] = 'block_fp8_mixed'
    core: Literal['native_lanes_v1'] = 'native_lanes_v1'
    # Default preserves the 70-g M1-only policy. The explicit value 16 selects
    # the existing multirow contract; complete-model quality is a separate gate.
    a8_max_rows: Literal[0, 1, 16] = 1
    reduce: Literal['sequential', 'neumaier_isa'] = 'neumaier_isa'
    post: Literal['vendor', 'fused_swa', 'fused_all'] = 'fused_swa'
    post_max_rows: Literal[1, 16] = 1


@dataclass(frozen=True)
class Norm:
    impl: Literal['vendor', 'fp32', 'fp32_grid24_quant'] = 'fp32_grid24_quant'


@dataclass(frozen=True)
class Attention:
    swa: Literal['vendor', 'fp32_fast', 'fp32_av_hoist', 'fp32_batch', 'fp32_batch_quad'] = 'fp32_av_hoist'
    full: Literal['vendor_sdpa'] = 'vendor_sdpa'


@dataclass(frozen=True)
class Router:
    post: Literal['vendor', 'vector_top8'] = 'vector_top8'


@dataclass(frozen=True)
class GateUp:
    impl: Literal['legacy', 'tpc_folded', 'tpc_folded_scale_tail'] = 'tpc_folded_scale_tail'


@dataclass(frozen=True)
class Down:
    impl: Literal['legacy', 'tpc_vector'] = 'tpc_vector'


@dataclass(frozen=True)
class Combine:
    sum_dtype: Literal['fp32', 'bf16'] = 'bf16'


@dataclass(frozen=True)
class MoEDispatch:
    # Explicit row set. The prior "max_rows=8" shorthand hid the B2
    # regression; compact-only B1/B8 has not passed a full-model gate yet.
    compact_rows: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    # Opt-in static shapes; per-expert ownership is determined on device.
    grouped_rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class MoE:
    router: Router = Router()
    gate_up: GateUp = GateUp()
    down: Down = Down()
    combine: Combine = Combine()
    dispatch: MoEDispatch = MoEDispatch()


@dataclass(frozen=True)
class TPReduce:
    collective: Literal['baseline', 'all_gather_fp32_local_sum'] = 'all_gather_fp32_local_sum'
    fp32_max_rows: Literal[1, 16] = 1
    fp32_max_bytes: Literal[131072, 524288] = 131072


@dataclass(frozen=True)
class Decode:
    qkv: QKV = QKV()
    norm: Norm = Norm()
    attention: Attention = Attention()
    moe: MoE = MoE()
    tp_reduce: TPReduce = TPReduce()


@dataclass(frozen=True)
class Buckets:
    batch: tuple[int, ...] = (1,)
    tokens: tuple[int, ...] = (1,)


@dataclass(frozen=True)
class HostPlacement:
    numa_binding: Literal['none', 'local'] = 'none'
    visible_modules: tuple[int, ...] = tuple(range(8))


@dataclass(frozen=True)
class Runtime:
    # native_graph is admitted only by an explicit producer/runtime gate.
    # Parsing its name must not turn the old recorder into a new executor.
    executor: Literal['pytorch', 'legacy_native', 'native_graph'] = 'legacy_native'
    runner: Literal['legacy_hooks', 'native'] = 'legacy_hooks'
    native_enabled: bool = False
    batch_replay: bool = False
    prefill_chunk_tokens: Literal[512, 2048, 4096] = 512
    flights: int = 1
    # SDK graph flush interval, NOT the number of model layers.
    graph_layer_interval: Literal[70] = 70
    buckets: Buckets = Buckets()
    host: HostPlacement = HostPlacement()


@dataclass(frozen=True)
class Diagnostics:
    profile: bool = False
    layer_hashes: bool = False
    same_input_audit: bool = False


@dataclass(frozen=True)
class EngineConfig:
    schema_version: Literal[1] = 1
    model: Literal['mimo-v2.6-pro'] = 'mimo-v2.6-pro'
    parallel: Parallel = Parallel()
    precision_contract: Literal['fp32_arithmetic_v1'] = 'fp32_arithmetic_v1'
    decode: Decode = Decode()
    runtime: Runtime = Runtime()
    diagnostics: Diagnostics = Diagnostics()

    @classmethod
    def from_dict(cls, document):
        result = _read(cls, document, 'config')
        result.validate()
        return result

    def validate(self):
        # Also validate direct Python construction, not only JSON loaders.
        _read(type(self), asdict(self), 'config')
        d = self.decode
        if (d.qkv.impl == 'block_fp8_a16') != (d.qkv.a8_max_rows == 0):
            raise ConfigError('A16 QKV requires a8_max_rows=0; mixed QKV requires 1')
        if d.qkv.post != 'vendor' and d.attention.swa == 'vendor':
            raise ConfigError('Fused QKV post requires the explicit FP32 SWA consumer')
        if d.norm.impl == 'fp32_grid24_quant' and (
                d.qkv.post == 'vendor' or d.qkv.impl != 'block_fp8_mixed'):
            raise ConfigError('grid24 requires mixed QKV and fused QKV post')
        if d.moe.combine.sum_dtype == 'bf16' and d.tp_reduce.collective != 'all_gather_fp32_local_sum':
            raise ConfigError('BF16 MoE final sum requires FP32 all-gather/local sum')
        compact = d.moe.dispatch.compact_rows
        _rows(compact, 'decode.moe.dispatch.compact_rows', allow_empty=True)
        if compact and max(compact) > 1 and (
                max(compact) > 8 or d.moe.gate_up.impl == 'legacy' or d.moe.down.impl != 'tpc_vector'):
            raise ConfigError('compact rows require qualified folded GP/vector down and rows <= 8')
        if compact not in ((), (1,), (1, 8), tuple(range(1, 9))):
            raise ConfigError('This migration admits only the implemented compact selector row sets')
        grouped = d.moe.dispatch.grouped_rows
        _rows(grouped, 'decode.moe.dispatch.grouped_rows', allow_empty=True)
        if any(n not in (512, 513, 1024, 2048, 4096) for n in grouped):
            raise ConfigError('Grouped MoE supports explicitly admitted large token shapes only')
        if not 1 <= self.runtime.flights <= 16:
            raise ConfigError('runtime.flights must be 1..16')
        _rows(self.runtime.buckets.batch, 'runtime.buckets.batch')
        _rows(self.runtime.buckets.tokens, 'runtime.buckets.tokens')
        if self.runtime.executor == 'pytorch' and (self.runtime.native_enabled or self.runtime.batch_replay):
            raise ConfigError('PyTorch execution must not enable an external replay engine')
        if self.runtime.batch_replay and (self.runtime.runner != 'native' or
                self.runtime.executor != 'legacy_native' or self.runtime.buckets.tokens != (1,)):
            raise ConfigError('Named batch replay requires the owned runner and T1; verify is not admitted yet')
        if self.runtime.native_enabled and self.runtime.runner!='native':
            raise ConfigError('Native serving startup requires the owned runner')
        modules = self.runtime.host.visible_modules
        if len(modules) != self.parallel.tp or len(set(modules)) != len(modules) or any(m < 0 for m in modules):
            raise ConfigError('runtime.host.visible_modules requires one distinct module per TP rank')

    def to_dict(self):
        # Canonical JSON turns tuples into lists and detaches mutable data.
        return json.loads(json.dumps(asdict(self)))

    @property
    def arithmetic_fingerprint(self):
        document = self.to_dict()
        document.pop('diagnostics')
        document.pop('runtime')
        return hashlib.sha256(json.dumps(document, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()


def _rows(values, path, allow_empty=False):
    if (not values and not allow_empty) or any(type(n) is not int or n < 1 for n in values):
        raise ConfigError(f'{path} must contain positive integers')
    if tuple(sorted(set(values))) != values:
        raise ConfigError(f'{path} must be sorted and unique')


def _read(kind, value, path):
    if is_dataclass(kind):
        if type(value) is not dict:
            raise ConfigError(f'{path} must be an object')
        names = {f.name for f in fields(kind)}
        if set(value) - names:
            raise ConfigError(f'{path}: unknown fields {sorted(set(value) - names)}')
        hints = get_type_hints(kind)
        return kind(**{name: _read(hints[name], child, f'{path}.{name}')
                       for name, child in value.items()})
    origin = get_origin(kind)
    if origin is Literal:
        if not any(type(value) is type(choice) and value == choice for choice in get_args(kind)):
            raise ConfigError(f'{path}: expected one of {get_args(kind)}, received {value!r}')
        return value
    if origin is tuple:
        if type(value) not in (list, tuple):
            raise ConfigError(f'{path} must be an array')
        element, repeat = get_args(kind)
        assert repeat is Ellipsis
        return tuple(_read(element, v, f'{path}[{i}]') for i, v in enumerate(value))
    if type(value) is not kind:
        raise ConfigError(f'{path} must be {kind.__name__}')
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f'Duplicate configuration field: {key}')
        result[key] = value
    return result


def load_config(path):
    return EngineConfig.from_dict(json.loads(Path(path).read_text(),
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ConfigError(f'Invalid JSON constant: {value}'))))
