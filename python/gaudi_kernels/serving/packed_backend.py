"""Registered attention extension; ordinary requests retain qualified kernels."""
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm_gaudi.v1.attention.backends.hpu_attn import (
    HPUAttentionDiffKVBackend, HPUAttentionDiffKVImpl,
)

from gaudi_kernels.serving.executor.packed_attention import PackedAttentionMetadata, forward_packed


class PackedDiffKVImpl(HPUAttentionDiffKVImpl):
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, **kwargs):
        if isinstance(attn_metadata, PackedAttentionMetadata):
            if any(value is not None for value in kwargs.values()):
                raise ValueError('Packed attention output quantization is unsupported')
            return forward_packed(self, layer, query, key, value, attn_metadata, output)
        if any(value is not None for value in kwargs.values()):
            raise ValueError('HPU attention output quantization is unsupported')
        return super().forward(layer, query, key, value, kv_cache, attn_metadata, output=output)


class PackedDiffKVBackend(HPUAttentionDiffKVBackend):
    @classmethod
    def set_head_size_v(cls, head_size_v):
        HPUAttentionDiffKVBackend.set_head_size_v(head_size_v)

    @staticmethod
    def get_impl_cls():
        return PackedDiffKVImpl


QUALIFIED_PACKED_FORWARD = PackedDiffKVImpl.forward


class PackedFlashDiffKVBackend(PackedDiffKVBackend):
    @staticmethod
    def is_supported_on_current_device(**kwargs):
        return False


def register():
    # Explicit worker setup after the platform registry and before construction.
    register_backend(AttentionBackendEnum.TRITON_ATTN_DIFFKV,
                     'gaudi_kernels.serving.packed_backend.PackedDiffKVBackend')
    register_backend(AttentionBackendEnum.FLASH_ATTN_DIFFKV,
                     'gaudi_kernels.serving.packed_backend.PackedFlashDiffKVBackend')
