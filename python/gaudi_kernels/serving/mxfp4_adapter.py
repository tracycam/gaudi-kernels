# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit routing adapter for the pinned vLLM HPU MXFP4 method.

Derived from vllm_gaudi.ops.hpu_mxfp4._HPUMxfp4MoEMixin.apply_monolithic.
Keep the existing FP32 routing contract without rewriting imported source text.
This belongs to the operator integration, not to the bridge execution core.
"""


def install_routing(module):
    from gaudi_kernels.serving.executor.precision_runtime import route_dtype

    def apply_monolithic(self, layer, x, router_logits, **kwargs):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if self.model_type == 'gpt_oss':
            weights, ids = module.torch.topk(router_logits, layer.top_k, dim=-1)
            weights = module.F.softmax(weights, dim=-1, dtype=module.torch.float32)
        elif layer.use_grouped_topk or getattr(layer, 'custom_routing_function', None) is not None:
            weights, ids = module.select_experts_from_routed(layer, x, router_logits)
        else:
            weights = module.F.softmax(router_logits, dim=1, dtype=module.torch.float32)
            weights, ids = module.torch.topk(weights, layer.top_k, dim=-1)
            weights /= weights.sum(dim=-1, keepdim=True)
        weights = weights.to(route_dtype(x))
        if not (layer.use_grouped_topk or getattr(layer, 'custom_routing_function', None) is not None):
            ids = ids.to(module.torch.int32)
            weights = weights.to(route_dtype(x))
        if layer.moe_config.dp_size > 1:
            metadata = module.get_hpu_dp_metadata()
            hidden = metadata.hidden_states_across_dp if metadata is not None else None
            x = module.dispatch_tensor(x, hidden, layer.moe_config.is_sequence_parallel)
            selected = metadata.topk_ids_across_dp if metadata is not None else None
            ids = module.dispatch_tensor(ids, selected, layer.moe_config.is_sequence_parallel)
            routed = metadata.topk_weights_across_dp if metadata is not None else None
            weights = module.dispatch_tensor(weights, routed, layer.moe_config.is_sequence_parallel)
        ids = ids.view(-1, ids.shape[-1])
        weights = weights.view(-1, weights.shape[-1])
        output = layer.moe_op(x, ids, weights, permuted_weights=True,
                              activation=module._normalize_moe_activation(layer.activation))
        shape = (output.size(0), *input_shape[1:]) if layer.moe_config.dp_size > 1 else input_shape
        return output if tuple(output.shape) == tuple(shape) else output.view(*shape)

    module._HPUMxfp4MoEMixin.apply_monolithic = apply_monolithic
