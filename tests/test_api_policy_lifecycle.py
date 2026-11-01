# SPDX-License-Identifier: Apache-2.0
"""Validation toggles cannot bypass the production policy lifecycle."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gaudi_kernels.engine.config import EngineConfig, ConfigError
from gaudi_kernels.engine.selection import PolicySelection
from tools.validation.executor.api_worker_extension import ApiWorkerExtension


class ApiPolicyLifecycleTests(unittest.TestCase):
    def test_grouped_toggle_delegates_once_and_does_not_mutate_selection(self):
        worker=ApiWorkerExtension()
        worker.rank=0
        worker.model_runner=SimpleNamespace(_policy_generation=4)
        selection=PolicySelection(EngineConfig())
        context=SimpleNamespace(selection=selection)
        with patch('gaudi_kernels.engine.context.context',return_value=context), \
                patch('gaudi_kernels.serving.policy.configure_policy',return_value={'moe_dispatch':{}}) as apply:
            result=worker.framework_grouped_prefill(True,8,32,4096)
            apply.assert_called_once()
            installed=PolicySelection.from_dict(apply.call_args.args[1]).engine.decode.moe.dispatch
        self.assertEqual(installed.grouped_rows,(512,))
        self.assertEqual(installed.grouped_slot_budget,32)
        self.assertEqual(installed.grouped_padded_row_budget,4096)
        self.assertEqual(selection.engine.decode.moe.dispatch.grouped_rows,())
        self.assertEqual(result['generation'],4)

    def test_invalid_budget_fails_before_lifecycle_or_device_work(self):
        context=SimpleNamespace(selection=PolicySelection(EngineConfig()))
        worker=ApiWorkerExtension()
        with patch('gaudi_kernels.engine.context.context',return_value=context), \
                patch('gaudi_kernels.serving.policy.configure_policy') as apply:
            with self.assertRaises(ConfigError):
                worker.framework_grouped_prefill(True,8,-1)
            with self.assertRaises(ConfigError):
                worker.framework_grouped_prefill(True,8,32,True)
            apply.assert_not_called()


if __name__=='__main__':
    unittest.main()
