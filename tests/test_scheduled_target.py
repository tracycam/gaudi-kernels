from types import SimpleNamespace
import unittest

import torch

from gaudi_kernels.engine.token_batch import RequestTokens, TokenBatch
from gaudi_kernels.serving.executor.packed_attention import forward_packed
from gaudi_kernels.serving.executor.scheduled_target import ScheduledKVBinding, token_slots


class ScheduledTargetTests(unittest.TestCase):
    def test_page_permutation_and_tail_slots(self):
        self.assertEqual(token_slots([3, 1], 6, 4), (12, 13, 14, 15, 4, 5))
        for pages, length in (([1], 5), ([1, 1], 5), ([-1], 1)):
            with self.assertRaises(ValueError):
                token_slots(pages, length, 4)

    def runner(self, pages):
        key, value = torch.zeros(32, 1, 4, dtype=torch.bfloat16), torch.zeros(32, 1, 3, dtype=torch.bfloat16)
        layer = SimpleNamespace(layer_name='attn', kv_cache=(key, value, None, None))
        table = SimpleNamespace(get_cpu_tensor=lambda: torch.tensor(pages))
        return SimpleNamespace(device='cpu',
            input_batch=SimpleNamespace(block_table=[table], req_id_to_index={'a': 0, 'b': 1}),
            kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=4), layer_names=['attn'])]),
            vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={'attn': layer})),
            _resolve_block=lambda block: block)

    def test_real_allocator_storage_is_updated_without_a_second_pool(self):
        runner = self.runner([[3, 1], [5, 2]])
        binding = ScheduledKVBinding(runner)
        batch = TokenBatch((RequestTokens('a', (1, 2), 0, 'prefill'), RequestTokens('b', (3,), 0, 'decode')))
        metadata = binding.prepare(batch)
        layer = runner.vllm_config.compilation_config.static_forward_context['attn']
        impl = SimpleNamespace(num_heads=2, num_kv_heads=1, head_size=4, head_size_v=3, scale=.5,
                               sliding_window=None, sinks=None, kv_sharing_target_layer_name=None, alibi_slopes=None)
        key_pointer = layer.kv_cache[0].data_ptr()
        q, k, v = torch.ones(3, 2, 4).bfloat16(), torch.ones(3, 1, 4).bfloat16(), torch.ones(3, 1, 3).bfloat16()
        forward_packed(impl, layer, q, k, v, metadata['attn'])
        binding.commit(metadata, (2, 1))
        self.assertEqual(layer.kv_cache[0].data_ptr(), key_pointer)
        self.assertTrue(torch.equal(layer.kv_cache[0][[12, 13, 20]], k))
        with self.assertRaises(ValueError):
            binding.commit(metadata, (2, 1))
        following = binding.prepare(TokenBatch((RequestTokens('a', (4,), 2, 'decode'),)))
        self.assertEqual(following['attn'].slot_mapping.tolist(), [14])
        binding.commit(following, (1,))

    def test_alias_and_speculation_are_rejected_before_kv_mutation(self):
        runner = self.runner([[3, 1], [3, 2]])
        binding = ScheduledKVBinding(runner)
        batch = TokenBatch((RequestTokens('a', (1,), 1, 'decode'), RequestTokens('b', (2,), 1, 'decode')))
        with self.assertRaisesRegex(ValueError, 'alias'):
            binding.prepare(batch)
        self.assertIsNone(binding.pending)
        with self.assertRaises(ValueError):
            binding.prepare(TokenBatch((RequestTokens('a', (1, 2), 0, 'verify'),)))
