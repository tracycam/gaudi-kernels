import importlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from gaudi_kernels.serving.module_bindings import ModuleBindings


class ModuleBindingsTests(unittest.TestCase):
    def test_existing_module_wraps_once_and_future_import_is_bound(self):
        name = '_gaudi_registration_fixture'
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, name + '.py').write_text('def call():\n    return 10\n')
            with patch.object(sys, 'path', [folder, *sys.path]), \
                    patch.object(sys, 'meta_path', list(sys.meta_path)):
                def bind(module):
                    old = module.call
                    module.call = lambda: old() + 1
                binding = ModuleBindings({name: bind})
                try:
                    binding.install()
                    module = importlib.import_module(name)
                    self.assertEqual(module.call(), 11)
                    binding.install()
                    self.assertEqual(module.call(), 11)
                    self.assertEqual(sys.meta_path.count(binding), 1)
                    with self.assertRaisesRegex(RuntimeError, 'Reload'):
                        importlib.reload(module)
                finally:
                    sys.modules.pop(name, None)
        module = ModuleType(name)
        module.call = lambda: 10
        with patch.dict(sys.modules, {name: module}), \
                patch.object(sys, 'meta_path', list(sys.meta_path)):
            binding = ModuleBindings({name: bind})
            binding.install()
            binding.install()
            self.assertEqual(module.call(), 11)

    def test_partial_failure_is_not_retried(self):
        name = '_gaudi_registration_failure'
        module = ModuleType(name)
        calls = []
        def fail(value):
            calls.append(value)
            raise ValueError('failure after mutation')
        with patch.dict(sys.modules, {name: module}), \
                patch.object(sys, 'meta_path', list(sys.meta_path)):
            binding = ModuleBindings({name: fail})
            with self.assertRaisesRegex(ValueError, 'mutation'):
                binding.install()
            with self.assertRaisesRegex(RuntimeError, 'fresh process'):
                binding.install()
            self.assertEqual(calls, [module])
            self.assertNotIn(binding, sys.meta_path)

    def test_module_replacement_and_incomplete_import_are_rejected(self):
        name = '_gaudi_registration_replacement'
        binding = ModuleBindings({name: lambda module: None})
        first = ModuleType(name)
        binding.apply(name, first)
        with self.assertRaisesRegex(RuntimeError, 'replaced'):
            binding.apply(name, ModuleType(name))
        second = ModuleType(name)
        second.__spec__ = SimpleNamespace(_initializing=True)
        with patch.dict(sys.modules, {name: second}):
            with self.assertRaisesRegex(RuntimeError, 'before importing'):
                ModuleBindings({name: lambda module: None}).install()

    def test_operator_bootstrap_import_is_inert_and_context_changes_reject(self):
        from gaudi_kernels.serving import bootstrap_hooks as hooks
        runtime = SimpleNamespace(startup=SimpleNamespace(inventory=False), has=lambda name: False)
        with patch.object(sys, 'meta_path', list(sys.meta_path)), \
                patch.object(hooks, '_REGISTRATION', None), patch.object(hooks, '_OWNER', None), \
                patch.object(hooks, 'execution_context', return_value=runtime), \
                patch.object(hooks, '_apply') as apply:
            name = 'vllm_gaudi.extension.ops'
            module = ModuleType(name)
            # Test the fixture independently of plugins imported by other tests.
            modules = {target: None for target in (
                'vllm.v1.executor.multiproc_executor',
                'vllm_gaudi.distributed.device_communicators.hpu_communicator',
                'vllm.model_executor.layers.fused_moe.router.gate_linear',
                'vllm_gaudi.ops.hpu_mxfp4')}
            modules[name] = module
            with patch.dict(sys.modules, modules):
                registration = hooks.install()
                self.assertIs(hooks.install(), registration)
                apply.assert_called_once_with(name, module)
            with patch.object(hooks, 'execution_context', return_value=object()):
                with self.assertRaisesRegex(RuntimeError, 'context changed'):
                    hooks.install()

    def test_reduced_loader_is_validation_only_and_preserves_other_weights(self):
        from tools.validation import partial_model
        name = 'vllm.model_executor.model_loader.default_loader'
        weights = [(key, object()) for key in (
            'model.layers.0.weight', 'model.layers.1.weight', 'model.layers.2.weight',
            'model.layers.20.weight', 'lm_head.weight', 'dflash.weight', 'layers.1.bias')]
        class Loader:
            def get_all_weights(self, marker):
                self.marker = marker
                yield from weights
        module = ModuleType(name)
        module.DefaultModelLoader = Loader
        with patch.dict(sys.modules, {name: module}), \
                patch.object(sys, 'meta_path', list(sys.meta_path)), \
                patch.object(partial_model, '_REGISTRATION', None), \
                patch.object(partial_model, '_LAYERS', None):
            partial_model.install(2)
            partial_model.install(2)
            loader = Loader()
            marker = object()
            self.assertEqual(list(loader.get_all_weights(marker)),
                             [weights[i] for i in (0, 1, 4, 5, 6)])
            self.assertIs(loader.marker, marker)
            with self.assertRaisesRegex(RuntimeError, 'layer count changed'):
                partial_model.install(3)


if __name__ == '__main__':
    unittest.main()
