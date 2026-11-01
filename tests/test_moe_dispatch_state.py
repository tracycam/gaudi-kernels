# SPDX-License-Identifier: Apache-2.0
"""Model ownership and capacity preflight without importing a device runtime."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from gaudi_kernels.engine.config import ConfigError, MoEDispatch
from gaudi_kernels.serving.executor.moe_dispatch_runtime import DispatchState, install_state


class Consumer:
    def set_moe_dispatch_state(self, state):
        self.state = state


class DispatchStateTests(unittest.TestCase):
    def test_models_do_not_share_policy_or_capture_counts(self):
        a, b = Consumer(), Consumer()
        pa = MoEDispatch(compact_rows=(1,))
        pb = replace(pa, grouped_rows=(512,), grouped_min_mme_rows=8, grouped_slot_budget=16)
        sa = install_state(SimpleNamespace(modules=lambda: [a]), pa)
        sb = install_state(SimpleNamespace(modules=lambda: [b]), pb)
        self.assertIs(a.state, sa)
        self.assertIs(b.state, sb)
        self.assertEqual(sa.select('auto', 512), 'broadcast')
        self.assertEqual(sb.select('auto', 512), 'grouped')
        self.assertEqual(sa.snapshot()['python_capture_calls_by_rows'], {'512': {'broadcast': 1}})
        self.assertEqual(sb.snapshot()['python_capture_calls_by_rows'], {'512': {'grouped': 1}})
        self.assertLessEqual(sb.plans[512].costs(512, 8, 384)['slots'], 16)

    def test_invalid_or_incompatible_installation_does_not_mutate_model(self):
        consumer = Consumer()
        model = SimpleNamespace(modules=lambda: [consumer])
        original = install_state(model, MoEDispatch())
        for policy in (MoEDispatch(grouped_slot_budget=True),
                       MoEDispatch(grouped_min_mme_rows=0),
                       MoEDispatch(grouped_rows=(8,))):
            with self.assertRaises(ConfigError):
                install_state(model, policy)
            self.assertIs(consumer.state, original)
        with self.assertRaises(ConfigError):
            install_state(model, MoEDispatch(compact_rows=(1,)), prepared=original)
        with self.assertRaises(ConfigError):
            install_state(SimpleNamespace(modules=lambda: []), MoEDispatch())

    def test_non_power_of_two_valid_rows_within_grouped_capacity_use_grouped_moe(self):
        state = DispatchState(MoEDispatch(compact_rows=(1,), grouped_rows=(512,)))
        self.assertIn(512, state.plans)
        self.assertNotIn(505, state.plans)
        self.assertTrue(state.supports_grouped(505))
        self.assertFalse(state.supports_grouped(128))
        self.assertFalse(state.supports_grouped(8))
        self.assertFalse(state.supports_grouped(1024))
        self.assertEqual(state.select('auto', 505), 'grouped')
        self.assertIn(505, state.plans)
        self.assertIs(state.plan_for(505), state.plans[505])
        self.assertTrue(state.plans[505].costs(505, 8, 384)['fits_budget'])
        self.assertEqual(state.select('auto', 128), 'broadcast')
        self.assertEqual(state.select('auto', 8), 'broadcast')
        self.assertEqual(state.select('auto', 1024), 'broadcast')

    def test_expert_moe_invokes_fused_grouped_forward_when_available(self):
        from unittest.mock import patch
        import torch
        from gaudi_kernels.moe_expert_dispatch import expert_moe
        from gaudi_kernels.serving.executor.grouped_moe_runtime import plan_for
        t, r, e = 505, 8, 384
        x = torch.empty((t, 6144), dtype=torch.bfloat16, device='meta')
        ids = torch.empty((t, r), dtype=torch.int32, device='meta')
        routing = torch.empty((t, r), dtype=torch.float32, device='meta')
        gp = torch.empty((e * 6144, 256), dtype=torch.uint8, device='meta')
        gs = torch.empty((e * 192, 512), dtype=torch.uint8, device='meta')
        down = torch.empty((e * 3072, 256), dtype=torch.uint8, device='meta')
        ds = torch.empty((e * 96, 512), dtype=torch.uint8, device='meta')
        lut = torch.empty((1, 512), dtype=torch.bfloat16, device='meta')
        calls = []
        fake_op = SimpleNamespace(
            grouped_forward=lambda *args: (calls.append(args), torch.empty((t, 6144), dtype=torch.float32, device='meta'))[1]
        )
        with patch.object(torch.ops, 'gaudi_expert_partition', fake_op):
            out = expert_moe(
                x, ids, routing, gp, gs, down, ds, lut, None,
                plan=plan_for(t), tpc=lambda *a, **k: None,
                decoder='k8', empty_mode=1, padding_mode=1, combine_mode='merged',
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(out.shape), (t, 6144))
        self.assertEqual(calls[0][8], [64, 505])
        self.assertEqual(calls[0][9], [384, 62])


if __name__ == '__main__':
    unittest.main()
