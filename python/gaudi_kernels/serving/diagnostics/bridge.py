"""Out-of-band framework ownership and graph coverage; never timed replay."""
import inspect
from pathlib import Path


def input_traversal(worker, enabled):
    """Benchmark-only A/B of installed bridge code vs a SHA-pinned old source."""
    import importlib.util
    import torch
    import habana_frameworks.torch.core as hc
    from habana_frameworks.torch.hpu import graphs
    from gaudi_kernels.engine.context import context
    if type(enabled) is not bool:
        raise ValueError('enabled must be bool')
    hc.mark_step(); torch.hpu.synchronize()
    state = getattr(worker, '_input_traversal_ab', None)
    if state is None:
        if not hasattr(graphs, '_input_hash'):
            raise RuntimeError('Installed bridge lacks the input traversal change')
        spec = importlib.util.spec_from_file_location('bridge_input_reference', context().path('input_reference', 'host'))
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        state = (graphs.wrapped_hpugraph_forward, reference.wrapped_hpugraph_forward)
        worker._input_traversal_ab = state
    graphs.wrapped_hpugraph_forward = state[0 if enabled else 1]
    return dict(rank=worker.rank, input_optimized=enabled)


def event_batching(worker, enabled, *, compiled=False):
    import torch
    import habana_frameworks.torch.core as hc
    from habana_frameworks.torch import _hpu_C
    if type(enabled) is not bool:
        raise ValueError('enabled must be bool')
    hc.mark_step(); torch.hpu.synchronize()
    count = 0
    for program, _ in getattr(worker.model_runner, '_step_graph_programs', {}).values():
        for cached in inspect.getclosurevars(program).nonlocals.get('cache', {}).values():
            graph = getattr(cached, 'graph', None)
            if graph is not None and _hpu_C.native_replay_stats(graph.hpu_graph)['ready']:
                setter = _hpu_C.native_replay_compiled_regions if compiled else _hpu_C.native_replay_event_batching
                setter(graph.hpu_graph, enabled)
                count += 1
    return dict(rank=worker.rank, enabled=enabled, compiled=compiled, graphs=count)


def snapshot(worker):
    from collections import Counter
    from dataclasses import fields, is_dataclass
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
    dependencies = []
    for cached in cache.values():
        graph = getattr(cached, 'graph', None)
        if graph is not None and hasattr(_hpu_C, 'native_replay_stats'):
            plans.append(_hpu_C.native_replay_stats(graph.hpu_graph))
            if hasattr(_hpu_C, 'native_dependency_diagnostics'):
                dependencies.append(dict(plan=len(plans)-1,
                    commands=_hpu_C.native_dependency_diagnostics(graph.hpu_graph)))
    step_plans = []
    input_tree_types = []
    def count_types(value, counts):
        counts[type(value).__module__+'.'+type(value).__qualname__] += 1
        if isinstance(value, dict):
            for pair in value.items():
                count_types(pair, counts)
        elif isinstance(value, (tuple, list)):
            for item in value:
                count_types(item, counts)
        elif is_dataclass(value):
            for field in fields(value):
                count_types(getattr(value, field.name), counts)
    for program, _ in getattr(worker.model_runner, '_step_graph_programs', {}).values():
        program_cache = inspect.getclosurevars(program).nonlocals.get('cache', {})
        for cached in program_cache.values():
            types = Counter()
            count_types(cached.graph_inputs, types)
            input_tree_types.append(dict(types))
            graph = getattr(cached, 'graph', None)
            if graph is not None and hasattr(_hpu_C, 'native_replay_stats'):
                step_plans.append(dict(stats=_hpu_C.native_replay_stats(graph.hpu_graph),
                    commands=_hpu_C.native_dependency_diagnostics(graph.hpu_graph)))
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if len(line.split()) >= 6 and '.so' in line.split()[-1]})
    forbidden = {runtime.path(name, 'host') for name in ('replay', 'tensor_hold', 'batch_replay')
                 if runtime.has(name)}
    external = [path for path in libraries if path in forbidden or
                'libe1_' in path or 'libgkg' in path]
    drafter = getattr(worker.model_runner, 'drafter', None)
    draft_plans = []
    if hasattr(drafter, 'model'):
        try:
            draft_cache = inspect.getclosurevars(drafter.model.forward).nonlocals.get('cache', {})
        except (TypeError, ValueError, AttributeError):
            draft_cache = {}
        for cached in draft_cache.values():
            if getattr(cached, 'graph', None) is not None:
                draft_plans.append(cached.graph.execution_info())
    return dict(rank=worker.rank, torch=str(torch.__version__), bridge=ht.__file__,
                executor=runtime.selection.engine.runtime.executor,
                graph_cache_entries=len(cache), plans=plans,
                step_plans=step_plans,
                input_tree_types=input_tree_types,
                step_graph_calls=getattr(worker.model_runner, '_step_graph_calls', 0),
                input_layout_hits=getattr(worker.model_runner, '_prepared_decode_input_hits', 0),
                input_layout_rejection=getattr(worker.model_runner, '_prepared_decode_input_rejection', None),
                dependency_diagnostics=dependencies,
                draft_plans=draft_plans,
                drafter_calls=getattr(drafter, 'calls', 0),
                drafter_context_owners=len(getattr(getattr(drafter, 'context', None), 'owners', {})),
                libraries=libraries, external_executors=external,
                scope='Diagnostic snapshot outside timing; empty plans are not native coverage')
