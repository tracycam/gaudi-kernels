"""Causal device prequeue probe; gate duration is excluded, never a TPS claim."""
from time import perf_counter_ns
import ctypes


def configure(worker, library=None):
    import torch
    import torch.distributed as dist
    import habana_frameworks.torch as ht
    from vllm.distributed import get_tp_group
    runner = worker.model_runner
    if library is None:
        state = runner.__dict__.pop('_prequeue_probe', None)
        if state is None:
            return dict(rank=worker.rank, rows=[])
        runner._execute_model_generic = state['original']
        ht.core.mark_step(); torch.hpu.synchronize()
        return dict(rank=worker.rank, rows=state['rows'],
                    scope='Device event interval after optional gate; excludes input prep; not TPS')
    if hasattr(runner, '_prequeue_probe') or not runner._step_graph_enabled():
        raise ValueError('Probe requires idle, complete-step framework mode')
    torch.ops.load_library(library)
    api=ctypes.CDLL(None)
    api.probe_bind_compute.argtypes=[ctypes.c_uint]
    api.probe_finish.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    def check(code):
        if code: raise RuntimeError(f'Compute event probe status {code}')
    x = torch.ones(64, device='hpu', dtype=torch.float32)
    marker = torch.zeros(1, device='hpu', dtype=torch.float32)
    gate = ht.hpu.wrap_in_hpu_graph_func(torch.ops.gk_diagnostic_gate.delay,
                                       disable_tensor_cache=False, asynchronous=False)
    def align(x):
        dist.all_reduce(x, group=get_tp_group().device_group)
        return x + 0
    alignment = ht.hpu.wrap_in_hpu_graph_func(align, disable_tensor_cache=False, asynchronous=False)
    held = (gate(x),)
    ht.core.mark_step(); torch.hpu.synchronize()
    check(api.probe_bind_compute(torch.hpu.current_device()))
    held = (held[0], alignment(marker))
    ht.core.mark_step(); torch.hpu.synchronize()
    original = runner._execute_model_generic
    state = dict(original=original, rows=[], seen=0, held=held)
    schedule = (False, True, True, False, False, True, True, False)

    def execute(*args, **kwargs):
        metadata = args[2] if len(args)>2 else kwargs['attn_metadata']
        if metadata.is_prompt or runner.input_batch.num_reqs != 1:
            return original(*args, **kwargs)
        state['seen'] += 1
        index = state['seen']-17
        if not 0 <= index < len(schedule):
            return original(*args, **kwargs)
        ht.core.mark_step(); torch.hpu.synchronize()
        dist.barrier(group=get_tp_group().cpu_group)
        if schedule[index]:
            state['held'] = (gate(x), alignment(marker))
        check(api.probe_begin())
        before = perf_counter_ns()
        result = original(*args, **kwargs)
        ht.core.mark_step()
        check(api.probe_end())
        enqueue_ns = perf_counter_ns()-before
        timing=(ctypes.c_uint64*2)()
        check(api.probe_finish(timing))
        state['rows'].append(dict(index=index, gated=schedule[index], prequeued=bool(timing[0]),
                                 enqueue_ms=enqueue_ns/1e6, device_ms=timing[1]/1e6,
                                 complete_step=runner._step_graph_sample is not None))
        return result
    runner._execute_model_generic = execute
    runner._prequeue_probe = state
    return dict(rank=worker.rank, status='ARMED')
