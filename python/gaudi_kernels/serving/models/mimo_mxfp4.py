"""MiMo TP8 expert weight/layout adapter; execution dispatch is shared separately."""
import json
import os
import time
import numpy as np
import torch
import habana_frameworks.torch.core as htcore
import gaudi_kernels.serving.executor.native_ops as native_ops
from gaudi_kernels.serving.executor.packing import pack, unpack

class NativeExpertTP(torch.nn.Module):

    def __init__(self, layer):
        super().__init__()
        for (key, src) in [('gp', 'w13_weight'), ('gs', 'w13_weight_scale'), ('dp', 'w2_weight'), ('ds', 'w2_weight_scale')]:
            self.register_buffer(key, getattr(layer, src).data)
        (table, directions) = native_ops.constants(self.gp.device)
        self.register_buffer('table', table)
        self.register_buffer('directions', directions)

    def forward(self, x, topk_ids, topk_weights, permuted_weights=True, activation='silu'):
        assert x.ndim == 2 and x.shape[-1] == 6144 and (topk_ids.ndim == 2)
        assert 1 <= topk_ids.shape[-1] <= 384 and topk_weights.shape == topk_ids.shape and (topk_ids.shape[0] == x.shape[0])
        assert activation == 'silu'
        from gaudi_kernels.serving.executor.moe_dispatch_runtime import forward
        from gaudi_kernels.serving.diagnostics import boundary_hashes as boundary
        output = forward(x, topk_ids, topk_weights, self.gp, self.gs,
                         self.dp, self.ds, self.table, self.directions)
        if boundary.active():
            return boundary.emit(self._boundary_layer_index, 'moe_local', output)
        return output


def process_weights(self, layer):
    start = time.monotonic()
    assert not self.moe.has_bias and layer.global_num_experts == layer.local_num_experts == 384
    assert tuple(layer.w13_weight.shape) == (384, 512, 3072), tuple(layer.w13_weight.shape)
    assert tuple(layer.w2_weight.shape) == (384, 6144, 128), tuple(layer.w2_weight.shape)
    assert layer.moe_config.dp_size == 1 and layer.moe_config.tp_size == 8
    metadata = []
    for (wn, sn) in [('w13_weight', 'w13_weight_scale'), ('w2_weight', 'w2_weight_scale')]:
        weight = getattr(layer, wn)
        scale = getattr(layer, sn)
        device = weight.device
        cpu_w = weight.detach().cpu().numpy()
        cpu_s = scale.detach().cpu().numpy()
        (e, n, k2) = cpu_w.shape
        k = k2 * 2
        (scale_min, scale_max) = (int(cpu_s.min()), int(cpu_s.max()))
        if scale_min < 8 or scale_max > 244:
            raise ValueError('Checkpoint contains scale values outside the certified model kernel range')
        packed = np.empty((e * (n // 512) * k, 256), np.uint8)
        scales = np.empty((e * (n // 512) * (k // 32), 512), np.uint8)
        for expert in range(e):
            (w, s) = pack(cpu_w[expert], cpu_s[expert])
            if expert in (0, e - 1):
                (pw, ps) = unpack(w, s)
                assert np.array_equal(pw, cpu_w[expert]) and np.array_equal(ps, cpu_s[expert])
            packed[expert * (n // 512) * k:(expert + 1) * (n // 512) * k] = w.reshape(-1, 256)
            scales[expert * (n // 512) * (k // 32):(expert + 1) * (n // 512) * (k // 32)] = s.reshape(-1, 512)
        setattr(layer, wn, torch.nn.Parameter(torch.from_numpy(packed).to(device), requires_grad=False))
        setattr(layer, sn, torch.nn.Parameter(torch.from_numpy(scales).to(device), requires_grad=False))
        htcore.mark_step()
        torch.hpu.synchronize()
        metadata.append(dict(name=wn, original_shape=[e, n, k2], packed_shape=list(packed.shape), bytes=int(packed.nbytes + scales.nbytes), scale_min=scale_min, scale_max=scale_max))
        del weight, scale, cpu_w, cpu_s, packed, scales
    layer.moe_op = NativeExpertTP(layer)
    layer._native_mxfp4 = True
    print('NATIVE_PACK ' + json.dumps(dict(pid=os.getpid(), seconds=time.monotonic() - start, weights=metadata)), flush=True)

def install(module):
    from gaudi_kernels.serving.executor.router_single_group import install as install_router
    install_router()
    print('ROUTER_SINGLE_GROUP_ENABLED decode_only=1', flush=True)
    module._HPUMxfp4MoEMixin.process_weights_after_loading = process_weights
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    base_final_reduce = MoERunner._maybe_reduce_final_output

    def final_reduce(self, states, trunc_size, output_is_reduced=None):
        native = getattr(self.routed_experts, '_native_mxfp4', False)
        reduced = self._fused_output_is_reduced if output_is_reduced is None else output_is_reduced
        from gaudi_kernels.serving.executor.moe_sum_bf16_runtime import final_scope
        with final_scope(self, states, trunc_size, reduced, native) as exact_final_scope:
            out = base_final_reduce(self, states, None if exact_final_scope else trunc_size, output_is_reduced)
        return out.to(torch.bfloat16) if native and out.dtype != torch.bfloat16 else out
    MoERunner._maybe_reduce_final_output = final_reduce
    print('NATIVE_MXFP4_INSTALLED ' + str(os.getpid()), flush=True)
