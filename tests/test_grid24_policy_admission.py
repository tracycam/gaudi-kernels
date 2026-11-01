from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from gaudi_kernels import vllm_norm_grid24 as grid


class Grid24PolicyAdmissionTests(unittest.TestCase):
    def test_small_batch_policy_admits_1_to_16_and_decode_policy_keeps_m1(self):
        layer=SimpleNamespace(input_layernorm=SimpleNamespace(
            weight=torch.empty(6144,device='meta',dtype=torch.bfloat16),variance_epsilon=1e-6),
            self_attn=SimpleNamespace())
        state=SimpleNamespace(policy='small_batch_a8_bf16_fp32',audit_enabled=False)
        with patch.object(grid,'_static',return_value=None),patch.object(grid.post,'_POLICY','fused'), \
                patch.object(grid.block,'_active',state),patch.object(grid.norm,'get_norm_policy',return_value='fp32'), \
                patch.object(grid.post,'_fallback_reason',return_value=None):
            for m in (1, 2, 4, 8, 12, 16):
                x=torch.empty(m,6144,device='meta',dtype=torch.bfloat16)
                pos=torch.empty(1,m,device='meta',dtype=torch.int32)
                self.assertIsNone(grid._reason(layer,pos,x,x))
            too_many=torch.empty(17,6144,device='meta',dtype=torch.bfloat16)
            self.assertEqual(grid._reason(layer,pos,too_many,too_many),'input_m1_bf16_6144')
            state.policy='decode_a8_bf16_fp32'
            x1=torch.empty(1,6144,device='meta',dtype=torch.bfloat16)
            self.assertIsNone(grid._reason(layer,pos,x1,x1))
            rows2=torch.empty(2,6144,device='meta',dtype=torch.bfloat16)
            self.assertEqual(grid._reason(layer,pos,rows2,rows2),'input_m1_bf16_6144')
            state.policy='bf16_fp32'
            self.assertEqual(grid._reason(layer,pos,x1,x1),'block_policy')
