"""Reversible, process-private precision policy applied before graph capture."""
from gaudi_kernels.engine.context import context as execution_context
import os
import torch
import habana_frameworks.torch.core as hc
import gaudi_kernels.serving.executor.precision_ops as precision_ops
COUNTS = {}

def hit(name):
    COUNTS[name] = COUNTS.get(name, 0) + 1

def enabled(name):
    if name not in ('FP8', 'GATE', 'ROUTER'):
        raise ValueError('Unknown precision scope')
    return True

def route_dtype(x):
    return torch.float32 if enabled('ROUTER') else x.dtype

def install(module):
    name = module.__name__
    if name.endswith('extension.ops'):
        original = module.apply_fp8_linear_hpu

        def linear(input, weight, weight_scale, input_scale=None, bias=None, trans_B=True):
            if not enabled('FP8'):
                return original(input, weight, weight_scale, input_scale, bias, trans_B)
            hit('fp8_f32_epilogue')
            if input_scale is None:
                (qx, sx) = module.dynamic_quant(input)
            else:
                qx = torch.ops.hpu.cast_to_fp8_v2(input, 1.0 / input_scale, False, False, torch.float8_e4m3fn)[0]
                sx = input_scale
            y = torch.ops.hpu.fp8_gemm_v2(qx, False, weight, trans_B, None, torch.float32, sx.float(), weight_scale.float(), None, False)
            if bias is not None:
                y = y + bias.float()
            return y.to(input.dtype)
        module.apply_fp8_linear_hpu = linear
    elif name.endswith('router.gate_linear'):
        original = module.GateLinear.forward

        def gate(self, x):
            if not enabled('GATE'):
                return original(self, x)
            if self.out_dtype == torch.float32 and self.weight.dtype == torch.bfloat16 and (x.dtype == torch.bfloat16):
                hit('router_mme_f32_output')
                y = precision_ops.mm(x.clone(), self.weight)
                bias = self.bias
                if bias is not None and (not self.skip_bias_add):
                    y = y + bias.float()
                return (y, bias if self.skip_bias_add else None)
            return original(self, x)
        module.GateLinear.forward = gate
    elif name.endswith('hpu_communicator'):
        original = module.HpuCommunicator.all_reduce

        def reduce(self, x):
            policy = execution_context().collective_mode
            max_rows = execution_context().selection.engine.decode.tp_reduce.fp32_max_rows
            max_bytes = execution_context().selection.engine.decode.tp_reduce.fp32_max_bytes
            if policy == 'off' or x.dtype not in (torch.bfloat16, torch.float32):
                return original(self, x)
            gather = policy == 'gather' or (policy in ('auto', 'auto_fp32') and x.dtype == torch.bfloat16 and (x.numel() * x.element_size() <= 128 * 1024)) or (policy == 'auto_fp32' and x.dtype == torch.float32 and x.ndim == 2 and x.shape[1] == 6144 and 1 <= x.shape[0] <= max_rows and x.numel() * x.element_size() <= max_bytes)
            if gather and x.ndim == 2 and (x.shape[1] % 128 == 0):
                hit('bf16_gather_f32_sum' if x.dtype == torch.bfloat16 else 'f32_gather_f32_sum')
                import torch.distributed as dist
                g = torch.empty((x.shape[0] * self.world_size, x.shape[1]), dtype=x.dtype, device=x.device)
                hc.mark_step()
                dist.all_gather_into_tensor(g, x, group=self.device_group)
                from gaudi_kernels.serving.executor.moe_sum_bf16_runtime import select_bf16_output
                final_bf16 = x.dtype == torch.float32 and select_bf16_output(x, self.world_size)
                if final_bf16:
                    hit('moe_final_f32_ag_bf16_sum')
                return precision_ops.sum_ranks(g, self.world_size, x.dtype == torch.bfloat16 or final_bf16)
            if x.dtype == torch.bfloat16:
                hit('promoted_f32_all_reduce')
                return original(self, x.float()).to(x.dtype)
            return original(self, x)
        module.HpuCommunicator.all_reduce = reduce
