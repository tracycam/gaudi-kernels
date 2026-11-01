"""Explicit, once-per-module compatibility registration for pinned plugins.

This is an import adapter, not an execution or graph API. Remove individual
bindings as the plugin acquires public registration points.
"""
import importlib.abc
import importlib.machinery
import sys


class ModuleBindings(importlib.abc.MetaPathFinder):
    def __init__(self, bindings):
        self.bindings = dict(bindings)
        self._installed = {}
        self._failed = None

    def apply(self, name, module):
        if self._failed is not None:
            raise RuntimeError('Compatibility registration failed; use a fresh process') from self._failed
        previous = self._installed.get(name)
        if previous is module:
            return
        if previous is not None:
            raise RuntimeError('Registered plugin module was replaced; use a fresh process')
        try:
            self.bindings[name](module)
        except BaseException as error:
            # Partial installation must never be retried and wrap calls twice.
            self._failed = error
            raise
        self._installed[name] = module

    def install(self):
        if self._failed is not None:
            raise RuntimeError('Compatibility registration failed; use a fresh process') from self._failed
        for name in self.bindings:
            module = sys.modules.get(name)
            if module is not None:
                if getattr(getattr(module, '__spec__', None), '_initializing', False):
                    raise RuntimeError('Initialize bindings before importing plugin modules')
                self.apply(name, module)
        if self not in sys.meta_path:
            sys.meta_path.insert(0, self)
        return self

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.bindings:
            return None
        if target is not None:
            raise RuntimeError('Reload of a registered plugin module is unsupported')
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        original, owner = spec.loader, self

        class Loader(importlib.abc.Loader):
            def create_module(self, module_spec):
                return original.create_module(module_spec)

            def exec_module(self, module):
                original.exec_module(module)
                owner.apply(fullname, module)

        spec.loader = Loader()
        return spec
