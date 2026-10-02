"""Explicit opt-in adapter for the pinned vllm-gaudi FP8 Linear lifecycle.

No import-time monkeypatch and no runtime graph/stream/device management. A
deployment selects the returned method class for block-FP8 Linear layers only.
The normal vLLM loader performs TP slicing; this adapter replaces its post-load
prepack and apply. MoE, ordinary FP8 and attention/KV management are untouched.
"""


def make_vllm_block_fp8_method(*, activation, scale_math='fp32', keep_cpu_oracle=False):
    """Return a block-only quant method using the upstream unwrapped loader.

    `OrigFp8LinearMethod` is the explicit upstream alias in pinned
    vllm_gaudi.ops.hpu_fp8. Calling its create_weights avoids that module's Gaudi
    loader wrapper, which already halves FP8 values and doubles scales. Running
    both the wrapper and our prepack would silently adapt twice.
    """
    import torch
    from dataclasses import replace
    from vllm_gaudi.ops.hpu_fp8 import OrigFp8LinearMethod
    from .block_fp8 import prepare_block_fp8, linear_block_fp8, PreparedBlockFP8
    if activation not in ('bf16', 'per_block_fp8') or scale_math not in ('fp32', 'bf16'):
        raise ValueError('explicit supported block FP8 arithmetic policies required')
    if activation == 'per_block_fp8' and scale_math != 'fp32':
        raise ValueError('W8A8 requires FP32 block scale/reduction')

    class BlockFP8LinearMethod(OrigFp8LinearMethod):
        def __init__(self, quant_config):
            super().__init__(quant_config)
            if (not self.block_quant or not quant_config.is_checkpoint_fp8_serialized
                    or tuple(quant_config.weight_block_size) != (128, 128)):
                raise ValueError('adapter only supports serialized block128 FP8 Linear')

        def create_weights(self, *args, **kwargs):
            # Do not call the Gaudi subclass's range-converting loader wrapper.
            return OrigFp8LinearMethod.create_weights(self, *args, **kwargs)

        def process_weights_after_loading(self, layer):
            if hasattr(layer, '_gk_block_fp8'):
                raise ValueError('block FP8 preparation must run exactly once')
            if hasattr(layer, '_hpu_orig_M') or getattr(layer, 'updated_fp8_weight', False):
                raise ValueError('already postprocessed Gaudi weight is not a raw checkpoint shard')
            device = layer.weight.device
            # Readback/transfer is allowed only in this load-time lifecycle hook.
            original = layer.weight.detach().cpu().contiguous()
            scales = layer.weight_scale_inv.detach().cpu().contiguous()
            # Explicit diagnostic owner only. Plain CPU attributes deliberately
            # remain outside parameters/buffers and never become HBM caches.
            import os
            if keep_cpu_oracle:
                layer._gk_oracle_weight = original
                layer._gk_oracle_scales = scales
                layer._gk_oracle_layout = {
                    'source': 'original layer.weight/weight_scale_inv after upstream TP loader, before native-half prepack',
                    'layer': getattr(layer, 'prefix', '<unknown>'),
                    'tp_size': getattr(layer, 'tp_size', None),
                    'tp_rank': getattr(layer, 'tp_rank', None),
                    'output_partition_sizes': list(getattr(layer, 'output_partition_sizes', []) or []),
                    'local_weight_shape': list(original.shape), 'local_scale_shape': list(scales.shape),
                    'local_scale_row_rule': 'floor(local_output_row/128)',
                    'limitation': 'upstream global-checkpoint TP grouping is a separate loader contract; not inferred from arbitrary slicing'}
            prepared = prepare_block_fp8(original, scales).to(device)
            # Replacing parameters releases the loader's old persistent storage.
            layer.weight = torch.nn.Parameter(prepared.weight, requires_grad=False)
            layer.weight_scale_inv = torch.nn.Parameter(prepared.scales, requires_grad=False)
            layer.register_buffer('_gk_zero_bias', prepared.bias)
            layer._gk_block_fp8 = replace(prepared, weight=layer.weight,
                scales=layer.weight_scale_inv, bias=layer._gk_zero_bias)
            layer._gk_block_fp8_policy = (activation, scale_math)

        def apply(self, layer, x, bias=None):
            prepared = layer._gk_block_fp8
            if x.shape[-1] != prepared.k:
                raise ValueError('vLLM block FP8 local activation K mismatch')
            # Shapes are graph metadata. No .item(), mark_step or host decisions
            # based on device values are introduced between model operators.
            y = linear_block_fp8(x.reshape(-1, prepared.k), prepared,
                activation=activation, scale_math=scale_math, bias=bias)
            return y.reshape(*x.shape[:-1], prepared.n)

    return BlockFP8LinearMethod
