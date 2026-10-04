"""Out-of-band framework ownership and graph coverage; never timed replay."""
import inspect
from pathlib import Path


def snapshot(worker):
    import torch
    import habana_frameworks.torch as ht
    from habana_frameworks.torch import _hpu_C
    from gaudi_kernels.engine.context import context
    runtime = context()
    model = worker.model_runner.model
    try:
        cache = inspect.getclosurevars(model.forward).nonlocals.get('cache', {})
    except (TypeError, ValueError, AttributeError):
        cache = {}
    plans = []
    for cached in cache.values():
        graph = getattr(cached, 'graph', None)
        if graph is not None and hasattr(_hpu_C, 'native_replay_stats'):
            plans.append(_hpu_C.native_replay_stats(graph.hpu_graph))
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if len(line.split()) >= 6 and '.so' in line.split()[-1]})
    forbidden = {runtime.path(name, 'host') for name in ('replay', 'tensor_hold', 'batch_replay')
                 if runtime.has(name)}
    external = [path for path in libraries if path in forbidden or
                'libe1_' in path or 'libgkg' in path]
    return dict(rank=worker.rank, torch=torch.__version__, bridge=ht.__file__,
                executor=runtime.selection.engine.runtime.executor,
                graph_cache_entries=len(cache), plans=plans,
                libraries=libraries, external_executors=external,
                scope='Diagnostic snapshot outside timing; empty plans are not native coverage')
