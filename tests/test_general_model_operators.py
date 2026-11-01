from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from gaudi_kernels.engine.config import ConfigError, EngineConfig, MoEDispatch
from gaudi_kernels.engine import context as runtime
from gaudi_kernels.engine.selection import PolicySelection
from gaudi_kernels import norm_qkv_explicit
from gaudi_kernels.production_integration import BlockFP8Installation
from gaudi_kernels.serving import bootstrap, bootstrap_hooks, launch, ops_registry
from gaudi_kernels.serving.executor import moe_dispatch_runtime, moe_sum_bf16_runtime
from gaudi_kernels import vllm_norm, vllm_norm_grid24 as grid


class GeneralModelOperatorTests(unittest.TestCase):
    def tearDown(self):
        runtime._CONTEXT = None
        moe_sum_bf16_runtime._POLICY = 'baseline'
        moe_sum_bf16_runtime._COUNTS.clear()

    def test_engine_config_and_manifest_support_non_mimo_models_and_tp_sizes(self):
        cfg = EngineConfig.from_dict({
            'model': 'smollm2-135m',
            'parallel': {'tp': 1},
            'runtime': {'host': {'visible_modules': [0]}},
        })
        self.assertEqual(cfg.model, 'smollm2-135m')
        self.assertEqual(cfg.parallel.tp, 1)
        with self.assertRaises(ConfigError):
            EngineConfig.from_dict({'model': '   '})

        with tempfile.TemporaryDirectory() as folder:
            lib = Path(folder) / 'lib.so'
            lib.write_bytes(b'fixture')
            import hashlib
            digest = hashlib.sha256(b'fixture').hexdigest()
            with patch.object(runtime, 'PINS', {'native.torch': digest}):
                doc = {
                    'schema_version': 1,
                    'startup': {'run_dir': str(Path(folder) / 'run'), 'layers': 30},
                    'artifacts': {'native.torch': {'path': str(lib), 'sha256': digest}},
                    'selection': PolicySelection(cfg).to_dict(),
                }
                ctx = runtime.initialize(doc, directory=folder)
                self.assertEqual(ctx.startup.layers, 30)
                with patch.object(bootstrap, 'load_manifest', return_value=ctx), \
                        patch.dict('os.environ', {'GK_RUNTIME_MANIFEST': 'manifest.json'}), \
                        patch.object(bootstrap_hooks, 'install') as hook_install:
                    self.assertIs(bootstrap.start(), ctx)
                    hook_install.assert_called_once_with()

                manifest = Path(folder) / 'manifest.json'
                manifest.write_text('{}')
                placement = {'binding': 'none', 'workers': [{'module_id': 0}]}
                with patch.object(launch, 'load_manifest', return_value=ctx), \
                        patch.object(launch, ' SDK_ENVIRONMENT', {}, create=True), \
                        patch('gaudi_kernels.serving.host_placement.prepare', return_value=placement):
                    _, command, _ = launch.prepare_launch(
                        manifest=manifest,
                        plugin_source=Path(folder),
                        vllm_source=Path(folder),
                        module='vllm.entrypoints.cli.main',
                    )
                self.assertIn('gaudi_kernels.serving.worker.FrameworkHPUWorker', command)

    def test_block_fp8_installation_admits_general_qkv_shapes_by_default(self):
        state = BlockFP8Installation('qkv', 'decode_a8_bf16_fp32', None, None)
        for shape in ((576, 576), (3392, 6144), (4096, 4096), (5120, 5120)):
            layer = SimpleNamespace(
                prefix='model.layers.0.self_attn.qkv_proj',
                _gk_block_fp8=SimpleNamespace(n=shape[0], k=shape[1]),
            )
            self.assertEqual(state.choose(layer, 1), ('per_block_fp8', 'fp32'))
            self.assertEqual(state.choose(layer, 2), ('bf16', 'fp32'))

        pinned = BlockFP8Installation('qkv', 'decode_a8_bf16_fp32', (4096, 4096), None)
        other_layer = SimpleNamespace(
            prefix='model.layers.0.self_attn.qkv_proj',
            _gk_block_fp8=SimpleNamespace(n=3392, k=6144),
        )
        self.assertEqual(pinned.choose(other_layer, 1), ('bf16', 'fp32'))

    def test_norm_qkv_explicit_and_grid24_support_general_hidden_sizes(self):
        for h in (256, 512, 2048, 4096, 6144, 8192):
            for m in (1, 2, 4, 8, 12, 16):
                x = torch.empty(m, h, device='meta', dtype=torch.bfloat16)
                g = torch.empty(h, device='meta', dtype=torch.bfloat16)
                self.assertIsNone(norm_qkv_explicit.input_reason(
                    x, x, g, 1e-6, norm_policy='fp32', activation_policy='per_block_fp8'))
            x2 = torch.empty(2, h, device='meta', dtype=torch.bfloat16)
            g = torch.empty(h, device='meta', dtype=torch.bfloat16)
            self.assertEqual(
                norm_qkv_explicit.input_reason(
                    x2, x2, g, 1e-6, norm_policy='fp32', activation_policy='per_block_fp8', max_rows=1),
                'input_m1_bf16_6144',
            )
            for m_over in (17, 32, 64, 505, 512, 4096):
                x_over = torch.empty(m_over, h, device='meta', dtype=torch.bfloat16)
                self.assertEqual(
                    norm_qkv_explicit.input_reason(
                        x_over, x_over, g, 1e-6, norm_policy='fp32', activation_policy='per_block_fp8'),
                    'input_m1_bf16_6144',
                )

        for bad_h in (128, 576, 8448):
            x = torch.empty(1, bad_h, device='meta', dtype=torch.bfloat16)
            g = torch.empty(bad_h, device='meta', dtype=torch.bfloat16)
            self.assertEqual(
                norm_qkv_explicit.input_reason(
                    x, x, g, 1e-6, norm_policy='fp32', activation_policy='per_block_fp8'),
                'input_m1_bf16_6144',
            )

        layer = SimpleNamespace(
            input_layernorm=SimpleNamespace(
                weight=torch.empty(4096, device='meta', dtype=torch.bfloat16),
                variance_epsilon=1e-6,
            ),
            self_attn=SimpleNamespace(),
        )
        state = SimpleNamespace(policy='decode_a8_bf16_fp32', audit_enabled=False)
        with patch.object(grid, '_static', return_value=None), \
                patch.object(grid.post, '_POLICY', 'fused'), \
                patch.object(grid.block, '_active', state), \
                patch.object(grid.norm, 'get_norm_policy', return_value='fp32'), \
                patch.object(grid.post, '_fallback_reason', return_value=None):
            x4096 = torch.empty(1, 4096, device='meta', dtype=torch.bfloat16)
            pos = torch.empty(1, 1, device='meta', dtype=torch.int32)
            self.assertIsNone(grid._reason(layer, pos, x4096, x4096))
            state.policy = 'small_batch_a8_bf16_fp32'
            for m in (1, 2, 4, 8, 12, 16):
                xm = torch.empty(m, 4096, device='meta', dtype=torch.bfloat16)
                posm = torch.empty(1, m, device='meta', dtype=torch.int32)
                self.assertIsNone(grid._reason(layer, posm, xm, xm))
            for m_prefill in (17, 64, 505, 512, 4096):
                xp = torch.empty(m_prefill, 4096, device='meta', dtype=torch.bfloat16)
                posp = torch.empty(1, m_prefill, device='meta', dtype=torch.int32)
                self.assertEqual(grid._reason(layer, posp, xp, xp), 'input_m1_bf16_6144')

    def test_plan_a_routes_small_m_to_sram_w8a8_and_large_m_to_w8a16(self):
        from gaudi_kernels.block_fp8 import (
            linear_block_fp8,
            prepare_block_fp8,
        )

        state = BlockFP8Installation('qkv', 'small_batch_a8_bf16_fp32', None, None)
        layer = SimpleNamespace(
            prefix='model.layers.0.self_attn.qkv_proj',
            _gk_block_fp8=SimpleNamespace(n=3392, k=6144),
        )
        for m in (1, 2, 4, 8, 12, 16):
            self.assertEqual(state.choose(layer, m), ('per_block_fp8', 'fp32'))
        for m in (17, 32, 64, 256, 505, 512, 1024, 2048, 4096):
            self.assertEqual(state.choose(layer, m), ('bf16', 'fp32'))

        all_state = BlockFP8Installation('all_block128', 'small_batch_a8_bf16_fp32', None, None)
        mlp_layer = SimpleNamespace(
            prefix='model.layers.0.mlp.gate_up_proj',
            _gk_block_fp8=SimpleNamespace(n=4096, k=6144),
        )
        for m in (1, 8, 16, 64, 512):
            self.assertEqual(all_state.choose(mlp_layer, m), ('bf16', 'fp32'))
        self.assertEqual(all_state.choose(layer, 1), ('per_block_fp8', 'fp32'))
        self.assertEqual(all_state.choose(layer, 512), ('bf16', 'fp32'))

        torch.manual_seed(20261015)
        n, k = 256, 256
        g = k // 128
        w = (torch.randn(n, k, dtype=torch.float32) * 0.1).to(torch.float8_e4m3fn).contiguous()
        s = (torch.rand(n // 128, g, dtype=torch.float32) * 0.02 + 0.005).contiguous()
        prepared = prepare_block_fp8(w, s)
        x = torch.randn(16, k, dtype=torch.bfloat16).contiguous()
        with self.assertRaises(ValueError):
            linear_block_fp8(x, prepared, activation='per_block_fp8', row_tile=8)

    def test_moe_sum_bf16_and_dispatch_support_general_tp_and_hidden_sizes(self):
        cfg = EngineConfig.from_dict({
            'model': 'qwen3-moe',
            'parallel': {'tp': 4},
            'runtime': {'host': {'visible_modules': [0, 1, 2, 3]}},
        })
        runtime._CONTEXT = SimpleNamespace(selection=PolicySelection(cfg))
        moe_sum_bf16_runtime.set_policy('moe_final_bf16')
        runner = SimpleNamespace(moe_config=SimpleNamespace(
            is_sequence_parallel=False,
            skip_final_all_reduce=False,
            tp_size=4,
            ep_size=1,
        ))
        states = SimpleNamespace(
            dtype=torch.float32,
            device=SimpleNamespace(type='hpu'),
            ndim=2,
            shape=(1, 2048),
            numel=lambda: 2048,
            element_size=lambda: 4,
            is_contiguous=lambda: True,
            requires_grad=False,
        )
        with moe_sum_bf16_runtime.final_scope(runner, states, 2048, False, True) as active:
            self.assertTrue(active)
            self.assertFalse(moe_sum_bf16_runtime.select_bf16_output(states, 8))
            self.assertTrue(moe_sum_bf16_runtime.select_bf16_output(states, 4))

        dispatch_state = moe_dispatch_runtime.DispatchState(
            MoEDispatch(compact_rows=(1,)), top_k=4, num_experts=64)
        self.assertEqual(dispatch_state.top_k, 4)
        self.assertEqual(dispatch_state.num_experts, 64)

        runtime._CONTEXT = SimpleNamespace(
            selection=PolicySelection(cfg),
            path=lambda *args: '/tmp/fake.so',
            has=lambda name: False,
        )
        with patch.object(torch.ops, 'load_library'), \
                patch.object(torch.library, 'register_fake', side_effect=lambda *a, **kw: (lambda fn: fn)):
            from gaudi_kernels.serving.executor import batch_ops
        empty_x = torch.empty((0, 2048), dtype=torch.bfloat16)
        empty_ids = torch.empty((0, 4), dtype=torch.int32)
        empty_routing = torch.empty((0, 4), dtype=torch.float32)
        out = batch_ops.moe(empty_x, empty_ids, empty_routing, None, None, None, None, None, None)
        self.assertEqual(tuple(out.shape), (0, 2048))

    def test_ops_registry_prepares_general_model_operators(self):
        class FakeNorm(torch.nn.Module):
            def __init__(self, use_native=False):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(576, dtype=torch.bfloat16), requires_grad=True)
                self.variance_epsilon = 1e-5
                self._forward_method = self.forward_native if use_native else self.forward_oot

            def forward_oot(self, x, residual=None):
                return x

            def forward_native(self, x, residual=None):
                return x

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = FakeNorm(use_native=False)
                self.norm_native = FakeNorm(use_native=True)

        model = FakeModel()
        with patch.object(vllm_norm, '_target', FakeNorm), \
                patch.object(vllm_norm, '_original_forward', FakeNorm.forward_oot):
            report = ops_registry.prepare_model_operators(model, norm_policy='fp32')
            self.assertEqual(report['norm']['prepared']['matched'], 2)
            self.assertEqual(report['norm']['prepared']['dispatch_rebound'], 2)
            self.assertEqual(report['norm']['policy']['policy'], 'fp32')
            self.assertFalse(model.norm.weight.requires_grad)
            self.assertFalse(model.norm_native.weight.requires_grad)
            vllm_norm.set_norm_policy('vendor')


if __name__ == '__main__':
    unittest.main()
