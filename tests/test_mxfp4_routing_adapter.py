"""Routing/shape contract independent of HPU kernels or source rewriting."""
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from gaudi_kernels.serving.mxfp4_adapter import install_routing


class RoutingAdapterTest(unittest.TestCase):
    def run_case(self, model_type, grouped=False, multidimensional=False):
        module = NS(torch=torch, F=F, _HPUMxfp4MoEMixin=type('Method', (), {}),
                    _normalize_moe_activation=lambda value: value)
        chosen = (torch.tensor([[.123456, .876544]], dtype=torch.float32),
                  torch.tensor([[2, 3]], dtype=torch.int64))
        module.select_experts_from_routed = lambda *args: chosen
        selected = []
        def op(x, ids, weights, **kwargs):
            selected.append((ids.clone(), weights.clone(), kwargs, x))
            return x
        layer = NS(top_k=2, use_grouped_topk=grouped, activation='silu',
                   moe_config=NS(dp_size=1), moe_op=op)
        value = torch.ones((1, 1, 4) if multidimensional else (1, 4), dtype=torch.bfloat16)
        logits = torch.tensor([[.1, .2, .7, -.3]], dtype=torch.float32)
        with patch.dict(sys.modules, {'gaudi_kernels.serving.executor.precision_runtime':
                                     NS(route_dtype=lambda _: torch.float32)}):
            install_routing(module)
        method = module._HPUMxfp4MoEMixin()
        method.model_type = model_type
        result = method.apply_monolithic(layer, value, logits)
        self.assertEqual(result.shape, value.shape)
        if not multidimensional:
            self.assertIs(result, selected[0][3])
        ids, weights, options, _ = selected[0]
        if model_type == 'gpt_oss':
            expected_values, expected_ids = torch.topk(logits, 2, dim=-1)
            expected_weights = F.softmax(expected_values, dim=-1, dtype=torch.float32)
        elif grouped:
            expected_weights, expected_ids = chosen
        else:
            expected_weights, expected_ids = torch.topk(F.softmax(logits, dim=1, dtype=torch.float32), 2, dim=-1)
            expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
        torch.testing.assert_close(ids, expected_ids, check_dtype=False, rtol=0, atol=0)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(options, dict(permuted_weights=True, activation='silu'))

    def test_regular_and_grouped_routes_preserve_fp32(self):
        self.run_case('mimo')
        self.run_case('mimo', grouped=True)

    def test_gpt_oss_order_and_multidimensional_output(self):
        self.run_case('gpt_oss')
        self.run_case('mimo', multidimensional=True)


if __name__ == '__main__':
    unittest.main()
