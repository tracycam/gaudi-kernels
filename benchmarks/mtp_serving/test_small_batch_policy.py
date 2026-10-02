import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


class SmallBatchPolicy(unittest.TestCase):
    def test_explicit_a8_contract_for_all_verify_rows_without_changing_legacy(self):
        path=Path(__file__).resolve().parents[2]/'python/gaudi_kernels/production_integration.py'
        spec=importlib.util.spec_from_file_location('small_batch_policy_under_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        layer=SimpleNamespace(prefix='model.layers.1.self_attn.qkv_proj',_gk_block_fp8=SimpleNamespace(n=3392,k=6144))
        legacy=module.BlockFP8Installation('qkv','decode_a8_bf16_fp32',None,None)
        candidate=module.BlockFP8Installation('qkv','small_batch_a8_bf16_fp32',None,None)
        for rows in (1,2,3,4,8,12,16,17,64,513):
            self.assertEqual(legacy.choose(layer,rows),('per_block_fp8'if rows==1 else 'bf16','fp32'))
            self.assertEqual(candidate.choose(layer,rows),('per_block_fp8'if rows<=16 else 'bf16','fp32'))
        layer.prefix='model.layers.1.o_proj'
        self.assertEqual(candidate.choose(layer,4),('bf16','fp32'))
        layer.prefix='model.layers.1.qkv_proj';layer._gk_block_fp8.n=4096
        self.assertEqual(candidate.choose(layer,4),('bf16','fp32'))


if __name__=='__main__':unittest.main()
