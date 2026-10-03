# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gaudi_kernels.engine.token_batch import RequestTokens, TokenBatch
from gaudi_kernels.serving.executor.packed_target import PackedTargetExecutor


class PackedResultLifetimeTests(unittest.TestCase):
    def test_results_survive_a_decoder_reusing_its_output_buffers(self):
        class ReusingModel(torch.nn.Module):
            start_layer, end_layer = 0, 2
            aux_hidden_state_layers = ()

            def __init__(self):
                super().__init__()
                self.hidden_buffer = torch.zeros(1, 8, dtype=torch.bfloat16)
                self.logits_buffer = torch.zeros(1, 16)

            def forward(self, input_ids, positions):
                self.hidden_buffer.fill_(int(input_ids[0]))
                if self.aux_hidden_state_layers:
                    return self.hidden_buffer, [self.hidden_buffer + i*100 for i in self.aux_hidden_state_layers]
                return self.hidden_buffer

            def _set_aux_hidden_state_layers(self, layers):
                self.aux_hidden_state_layers = layers

            def compute_logits(self, hidden):
                self.logits_buffer.fill_(float(hidden[0, 0]) + .5)
                return self.logits_buffer

        class Impl:
            num_kv_heads, head_size, head_size_v = 1, 4, 3

        model = ReusingModel()
        runner = SimpleNamespace(device='cpu', model=SimpleNamespace(model=model,
            _rotary_prepare_cos_sin=None, flatten_positions=lambda x: x),
            vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16),
                compilation_config=SimpleNamespace(static_forward_context={'layer': SimpleNamespace(impl=Impl())})))
        context = ModuleType('vllm.forward_context')
        @contextmanager
        def forward_context(*args, **kwargs):
            yield
        context.set_forward_context = forward_context
        backend = ModuleType('gaudi_kernels.serving.packed_backend')
        backend.PackedDiffKVImpl = Impl
        root, htorch, core = [ModuleType(name) for name in
                             ('habana_frameworks', 'habana_frameworks.torch', 'habana_frameworks.torch.core')]
        root.torch, htorch.core = htorch, core
        core.mark_step = lambda: None
        with patch.dict(sys.modules, {'vllm.forward_context': context,
                'gaudi_kernels.serving.packed_backend': backend, 'habana_frameworks': root,
                'habana_frameworks.torch': htorch, 'habana_frameworks.torch.core': core}):
            # CPU substitute covers output ownership only, not HPU arithmetic.
            with patch.object(torch, 'hpu', SimpleNamespace(synchronize=lambda: None), create=True):
                executor = PackedTargetExecutor(runner, kv_capacity=4, feature_layers=(0, 1))
                first = executor.execute(TokenBatch((RequestTokens('r', (3,), 0, 'decode'),)))
                self.assertEqual(model.aux_hidden_state_layers, ())
                executor.commit(first, (1,))
                second = executor.execute(TokenBatch((RequestTokens('r', (7,), 1, 'decode'),)))
                self.assertTrue(torch.equal(first.hidden, torch.full((1, 8), 3, dtype=torch.bfloat16)))
                self.assertTrue(torch.equal(first.logits, torch.full((1, 16), 3.5)))
                self.assertTrue(torch.equal(second.hidden, model.hidden_buffer))
                self.assertNotEqual(first.hidden.data_ptr(), model.hidden_buffer.data_ptr())
                self.assertNotEqual(first.logits.data_ptr(), model.logits_buffer.data_ptr())
                self.assertEqual(first.feature_layers, (0, 1))
                self.assertTrue(torch.equal(first.features[0], torch.full((1, 8), 103, dtype=torch.bfloat16)))
                self.assertTrue(torch.equal(first.features[1], torch.full((1, 8), 203, dtype=torch.bfloat16)))
                executor.abort(second)
