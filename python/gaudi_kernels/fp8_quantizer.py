"""Explicit, loading-time constants for device-qualified FP8 quantizer candidates.

Construct/transfer once before capture. Keep this owner alive and its table
unchanged during graph replay. Importing this module does not access a device.
"""
from dataclasses import dataclass
import hashlib
import struct

_TABLE_SHA256 = "d470a4742864c0e7903ec196bc0d6bb8572e8e3e937d5b0819551f3c984e8b59"
_TABLE_WORDS = (
    0x43dfffff, 0x43de4379, 0x43dc8dc8, 0x43dadec6, 0x43d9364d, 0x43d79436, 0x43d5f85b, 0x43d4629b,
    0x43d2d2d3, 0x43d148df, 0x43cfc4a2, 0x43ce45fc, 0x43cccccd, 0x43cb58f6, 0x43c9ea5d, 0x43c880e5,
    0x43c71c71, 0x43c5bce8, 0x43c46231, 0x43c30c31, 0x43c1bacf, 0x43c06df5, 0x43bf258c, 0x43bde17b,
    0x43bca1af, 0x43bb6610, 0x43ba2e8b, 0x43b8fb0b, 0x43b7cb7c, 0x43b69fcc, 0x43b577e5, 0x43b453b9,
    0x43b33333, 0x43b21642, 0x43b0fcd6, 0x43afe6df, 0x43aed44b, 0x43adc50a, 0x43acb90f, 0x43abb049,
    0x43aaaaaa, 0x43a9a824, 0x43a8a8a8, 0x43a7ac2a, 0x43a6b29a, 0x43a5bbee, 0x43a4c817, 0x43a3d709,
    0x43a2e8ba, 0x43a1fd1b, 0x43a11422, 0x43a02dc3, 0x439f49f4, 0x439e68aa, 0x439d89d8, 0x439cad76,
    0x439bd37a, 0x439afbd9, 0x439a2689, 0x43995382, 0x439882b9, 0x4397b425, 0x4396e7bf, 0x43961d7c,
    0x43955555, 0x43948f40, 0x4393cb37, 0x43930930, 0x43924924, 0x43918b0b, 0x4390cede, 0x43901495,
    0x438f5c28, 0x438ea592, 0x438df0cb, 0x438d3dca, 0x438c8c8c, 0x438bdd08, 0x438b2f38, 0x438a8317,
    0x4389d89d, 0x43892fc5, 0x43888888, 0x4387e2e1, 0x43873ecb, 0x43869c3e, 0x4385fb37, 0x43855baf,
    0x4384bda1, 0x43842107, 0x438385df, 0x4382ec21, 0x438253c8, 0x4381bcd0, 0x43812735, 0x438092f1,
    0x43800000, 0x437edcb9, 0x437dbc08, 0x437c9de2, 0x437b823e, 0x437a6915, 0x4379525d, 0x43783e10,
    0x43772c22, 0x43761c90, 0x43750f50, 0x4374045b, 0x4372fba9, 0x4371f533, 0x4370f0f1, 0x436feedb,
    0x436eeeee, 0x436df120, 0x436cf56b, 0x436bfbc9, 0x436b0432, 0x436a0ea1, 0x43691b0d, 0x43682974,
    0x436739ce, 0x43664c14, 0x43656041, 0x4364764f, 0x43638e39, 0x4362a7f7, 0x4361c386, 0x4360e0e0,
    0x54826299,
)


@dataclass(frozen=True)
class PreparedFP8Quantizer:
    table: object
    variant: str = 'lut4'
    format_version: int = 1

    def validate(self, device=None):
        import torch
        if (self.format_version != 1 or self.variant not in ('lut1','lut4')
                or not torch.is_tensor(self.table) or self.table.dtype != torch.float32
                or tuple(self.table.shape) != (1,129) or not self.table.is_contiguous()
                or self.table.requires_grad):
            raise ValueError('version1 FP8 quantizer requires a contiguous immutable FP32 [1,129] table')
        if device is not None and self.table.device != device:
            raise ValueError('FP8 quantizer must already be on the activation device')
        return self

    def to(self, device):
        self.validate()
        return PreparedFP8Quantizer(self.table.to(device),self.variant,self.format_version)


def prepare_fp8_quantizer(variant='lut4'):
    """Preserve the existing per-token scale and two-rounding FP8 policy.

    The table comes from the finite-domain certificate, including its one-ULP
    correction at mantissa89. No weight preparation or activation is changed.
    This is explicit opt-in; no per-forward allocation/transfer is performed.
    """
    import torch
    if variant not in ('lut1','lut4'):raise ValueError('variant must be lut1 or lut4')
    assert hashlib.sha256(struct.pack('<129I',*_TABLE_WORDS)).hexdigest() == _TABLE_SHA256
    table=torch.tensor(_TABLE_WORDS,dtype=torch.int32).view(torch.float32).reshape(1,129)
    return PreparedFP8Quantizer(table,variant).validate()
