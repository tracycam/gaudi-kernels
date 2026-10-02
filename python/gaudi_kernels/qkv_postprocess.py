"""Explicit QKV postprocessing; the cache variant has scoped device gates.

Never selects an arithmetic boundary implicitly, mutates a KV cache, or builds
an extension on import. Existing index_copy_ remains the cache update mechanism.
"""
import torch


def qkv_postprocess_cache(qkv, cosine_sine_cache, positions, value_scale):
    """Direct layer-cache BF16-boundary candidate; never enables itself.

    qkv[M,3392] -> Q[M,3072], K[M,1,192], V[M,1,128]. The actual layer's
    BF16 cache[P,64] stores cos32 then sin32; positions index its rows. RoPE
    rounds both BF16 products then the BF16 sum; V uses BF16(RN32(scale)).
    Original BF16 QKV is a required boundary. Invalid positions return zero
    Q/K/V without an out-of-bounds access; qualified calls require valid rows.
    Long metadata relies on the pinned Lazy native-I32 representation and the
    native glue rejects I64. Multi-output capture and original-cache-write/SWA
    chains are device-qualified in docs/QKV-POSTPROCESS-DEVICE-20260928.md;
    production model dispatch and whole-model quality remain unqualified.
    """
    return torch.ops.gaudi_kernels._qkv_post_cache_bf16_v2(
        qkv,cosine_sine_cache,positions,float(value_scale))


def qkv_postprocess(qkv, cosine, sine, value_scale, *, rope_products, value_scale_math):
    """Return BF16 Q[M,16,192],K[M,1,192],V[M,1,128] for MiMo TP8.

    Inputs: BF16 qkv[M,3392], selected/expanded cos/sin[M,1,64].
    ``rope_products`` and ``value_scale_math`` must explicitly be 'bf16'/'f32'.
    'bf16' rounds RoPE's two products before adding, or rounds the V scalar
    before multiplying, respectively. None is yet vendor-device-qualified.
    """
    if rope_products not in ('bf16','f32') or value_scale_math not in ('bf16','f32'):
        raise ValueError('arithmetic must explicitly be bf16 or f32')
    return getattr(torch.ops.gaudi_kernels, '_qkv_post_'+rope_products+'_'+value_scale_math)(
        qkv, cosine, sine, float(value_scale))
