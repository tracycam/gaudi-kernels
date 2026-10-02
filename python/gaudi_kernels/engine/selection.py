"""Typed selection for the existing installer during equivalent migration.

Legacy spellings are emitted only as diagnostic labels. Production callers
send this structured document to workers; workers never parse a '+' string.
No module import, library load or arithmetic is performed here.
"""
from dataclasses import dataclass
from typing import Literal
from .config import ConfigError, EngineConfig

Reference = Literal['device', 'cpu_ocp_a8_bf16_fp32', 'cpu_native_a8_bf16_fp32',
                    'cpu_ocp_a8_fp32_v1', 'cpu_native_a8_fp32_v1']
_REFERENCES = ('device', 'cpu_ocp_a8_bf16_fp32', 'cpu_native_a8_bf16_fp32',
               'cpu_ocp_a8_fp32_v1', 'cpu_native_a8_fp32_v1')


@dataclass(frozen=True)
class PolicySelection:
    engine: EngineConfig
    reference: Reference = 'device'

    def __post_init__(self):
        if type(self.engine) is not EngineConfig or self.reference not in _REFERENCES:
            raise ConfigError('Invalid typed policy selection')
        self.engine.validate()
        from .dispatch import DispatchTable
        DispatchTable(self.engine).runtime_admission()

    @classmethod
    def from_dict(cls, document):
        if type(document) is not dict or set(document) != {'engine', 'reference'}:
            raise ConfigError('Policy selection requires exactly engine/reference')
        return cls(EngineConfig.from_dict(document['engine']), document['reference'])

    def to_dict(self):
        return {'engine': self.engine.to_dict(), 'reference': self.reference}

    @property
    def block_policy(self):
        if self.reference != 'device':
            return self.reference
        if self.engine.decode.qkv.a8_max_rows == 16:
            return 'small_batch_a8_bf16_fp32'
        return 'decode_a8_bf16_fp32' if self.engine.decode.qkv.impl == 'block_fp8_mixed' else 'bf16_fp32'
