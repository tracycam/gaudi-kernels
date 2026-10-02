import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
from gaudi_kernels import vllm_qkv_postprocess as post


class QKVPostRowTests(unittest.TestCase):
    def test_opt_in_rows_still_reject_prefill_and_wrong_position_ownership(self):
        device=SimpleNamespace(type='hpu')
        class Tensor:
            def __init__(self,shape,dtype):
                self.shape=shape;self.ndim=len(shape);self.dtype=dtype
                self.device=device;self.requires_grad=False
            def is_contiguous(self):return True
        fake=SimpleNamespace(Tensor=Tensor,bfloat16='bf16',int32='i32',int64='i64')
        metadata=SimpleNamespace(is_prompt=False,block_size=128)
        context=ModuleType('vllm.forward_context')
        context.get_forward_context=lambda:SimpleNamespace(attn_metadata=metadata)
        context.is_forward_context_available=lambda:True
        module=SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=SimpleNamespace(device=device)))
        with patch.object(post,'torch',fake),patch.object(post,'_static_reason',return_value=None), \
                patch.dict(sys.modules,{'vllm.forward_context':context}),patch.object(post,'_MAX_ROWS',1):
            x=Tensor((4,6144),'bf16');positions=Tensor((4,1),'i32')
            self.assertEqual(post._fallback_reason(module,positions,x),'hidden_shape')
            with patch.object(post,'_MAX_ROWS',16):
                self.assertIsNone(post._fallback_reason(module,positions,x))
                self.assertEqual(post._fallback_reason(module,Tensor((1,1),'i32'),x),'positions_shape')
                metadata.is_prompt=True
                self.assertEqual(post._fallback_reason(module,positions,x),'prompt')
                metadata.is_prompt=False
                self.assertEqual(post._fallback_reason(module,Tensor((17,1),'i32'),Tensor((17,6144),'bf16')),'hidden_shape')
