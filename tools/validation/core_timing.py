"""Validation-only EngineCore producer intervals; no per-token file I/O."""
import importlib.abc
import importlib.machinery
import sys
import time


def instrument(module):
    cls = module.EngineCore
    if getattr(cls, '_runtime_timing_installed', False):
        return
    original = cls.step
    def step(self):
        begin = time.perf_counter_ns()
        outputs, executed = original(self)
        end = time.perf_counter_ns()
        if executed:
            records = getattr(self, '_runtime_timing_records', None)
            if records is None:
                records = self._runtime_timing_records = []
            tokens = [{'request_id': output.request_id, 'tokens': list(output.new_token_ids)}
                      for batch in outputs.values() for output in batch.outputs]
            records.append({'begin_ns': begin, 'end_ns': end, 'tokens': tokens})
        return outputs, executed
    def take(self):
        records = getattr(self, '_runtime_timing_records', [])
        self._runtime_timing_records = []
        return records
    cls.step = step
    cls.runtime_timing_take = take
    cls._runtime_timing_installed = True


def install():
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != 'vllm.v1.engine.core':
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            original = spec.loader
            class Loader(importlib.abc.Loader):
                def create_module(self, spec):
                    return original.create_module(spec)
                def exec_module(self, module):
                    original.exec_module(module)
                    instrument(module)
            spec.loader = Loader()
            return spec
    sys.meta_path.insert(0, Finder())
