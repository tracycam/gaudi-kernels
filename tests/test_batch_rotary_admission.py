"""Exercise the actual wrapper's admission expression without an HPU context."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


class Tensor:
    def __init__(self, shape, contiguous=True):
        self.shape=shape;self.dtype='bf16';self.device=SimpleNamespace(type='hpu')
        self.requires_grad=False;self.contiguous=contiguous
    def is_contiguous(self):return self.contiguous
    def numel(self):return math.prod(self.shape)


class Rotary:
    head_size=192
    rotary_dim=64


class BatchRotaryTests(unittest.TestCase):
    def eligible(self, policy, rows, contiguous):
        path=Path(__file__).resolve().parents[1]/'python/gaudi_kernels/vllm_swa_rotary.py'
        tree=ast.parse(path.read_text())
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward')
        assignments=[]
        for node in fn.body:
            if isinstance(node,ast.If):break
            assignments.append(node)
        namespace={'policy':lambda:policy,'self':Rotary(),'HPURotaryEmbedding':Rotary,
            'torch':SimpleNamespace(Tensor=Tensor,bfloat16='bf16'),
            'positions':Tensor((rows,)),'query':Tensor((rows,3072),contiguous),
            'key':Tensor((rows,192),contiguous)}
        # Device identity is checked too; all tensors belong to the same HPU.
        namespace['query'].device=namespace['positions'].device
        namespace['key'].device=namespace['positions'].device
        exec(compile(ast.Module(body=assignments,type_ignores=[]),str(path),'exec'),namespace)
        return namespace['eligible']

    def test_strided_multirow_qkv_slices_use_owned_rope_restore_only_when_selected(self):
        for rows in (2,3,4,8,12,16,32):
            self.assertTrue(self.eligible('fp32_batch_quad',rows,False))
            self.assertFalse(self.eligible('fp32_av_hoist',rows,False))
        self.assertTrue(self.eligible('fp32_av_hoist',1,True))
        self.assertFalse(self.eligible('fp32_av_hoist',1,False))
        self.assertFalse(self.eligible('fp32_batch_quad',33,True))
        self.assertFalse(self.eligible('vendor',2,True))
