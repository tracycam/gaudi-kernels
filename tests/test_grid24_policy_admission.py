from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from gaudi_kernels import vllm_norm_grid24 as grid


class Grid24PolicyAdmissionTests(unittest.TestCase):
    def test_small_batch_policy_keeps_m1_and_preserves_multirow_rejection(self):
        layer=SimpleNamespace(input_layernorm=SimpleNamespace(
            weight=torch.empty(6144,device='meta',dtype=torch.bfloat16),variance_epsilon=1e-6),
            self_attn=SimpleNamespace())
        state=SimpleNamespace(policy='small_batch_a8_bf16_fp32',audit_enabled=False)
        with patch.object(grid,'_static',return_value=None),patch.object(grid.post,'_POLICY','fused'), \
                patch.object(grid.block,'_active',state),patch.object(grid.norm,'get_norm_policy',return_value='fp32'), \
                patch.object(grid.post,'_fallback_reason',return_value=None):
            x=torch.empty(1,6144,device='meta',dtype=torch.bfloat16)
            pos=torch.empty(1,1,device='meta',dtype=torch.int32)
            self.assertIsNone(grid._reason(layer,pos,x,x))
            rows=torch.empty(2,6144,device='meta',dtype=torch.bfloat16)
            self.assertEqual(grid._reason(layer,pos,rows,rows),'input_m1_bf16_6144')
            state.policy='bf16_fp32'
            self.assertEqual(grid._reason(layer,pos,x,x),'block_policy')
