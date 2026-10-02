"""Historical installer access exists only in the benchmark worker process."""
import importlib.abc
import importlib.machinery
import sys


def install():
    from gaudi_kernels.serving.bootstrap_hooks import Finder
    class ReferenceFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != 'vllm_gaudi.v1.worker.hpu_worker': return None
            spec = Finder().find_spec(fullname, path, target)
            if spec is None:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            original = spec.loader
            class Loader(importlib.abc.Loader):
                def create_module(self, module_spec): return original.create_module(module_spec)
                def exec_module(self, module):
                    original.exec_module(module)
                    from tools.validation.reference_installer import configure_reference
                    module.HPUWorker.production_policy_reference = configure_reference
            spec.loader = Loader()
            return spec
    sys.meta_path.insert(0, ReferenceFinder())
