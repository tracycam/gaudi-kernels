"""Bounded direct SDK capture, reusing the audited exporter, no Torch profiler."""
import ctypes
from pathlib import Path


def configure(worker, library, tag, enabled=True):
    import torch
    import habana_frameworks.torch.core as hc
    from gaudi_kernels.engine.context import context
    runner = worker.model_runner
    if not enabled:
        state = runner.__dict__.pop('_sdk_profile_state', None)
        if state is None:
            return dict(rank=worker.rank, status='not_armed')
        runner.sample_tokens = state['original']
        if state['started']:
            hc.mark_step(); torch.hpu.synchronize()
            finish(state)
        return dict(rank=worker.rank, path=str(state['path']), samples=state['samples'],
                    status='PASS' if state['samples'] == 3 else 'INCOMPLETE',
                    bytes=state.get('bytes'), entries=state.get('entries'))
    if hasattr(runner, '_sdk_profile_state'):
        raise RuntimeError('SDK profile already armed')
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in tag):
        raise ValueError('Invalid profile tag')
    api = ctypes.CDLL(str(Path(library).resolve(strict=True)))
    api.trace_device.argtypes = [ctypes.c_uint32]
    api.trace_required.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    api.trace_start.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    api.trace_stop.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
    api.synProfilerGetCurrentTimeNS.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    api.synProfilerAddCustomMeasurement.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    check(api.trace_device(torch.hpu.current_device()))
    required = ctypes.c_uint32()
    check(api.trace_required(ctypes.byref(required)))
    if required.value > 512*1024*1024:
        raise ValueError('SDK profile buffer exceeds 512MiB')
    storage = torch.empty(max(4096, required.value), dtype=torch.uint8, device='hpu')
    hc.mark_step(); torch.hpu.synchronize()
    path = context().startup.run_dir / f'{tag}-rank{worker.rank}.jsonl'
    if path.exists():
        raise FileExistsError(path)
    original = runner.sample_tokens
    state = dict(api=api, storage=storage, path=path, original=original,
                 seen=0, samples=0, started=False)

    def sample(*args, **kwargs):
        sched = runner.scheduler_output
        eligible = (sched is not None and not runner.warmup_mode and
                    runner.input_batch.num_reqs == 1 and not runner._is_prompt(0, sched))
        if eligible:
            state['seen'] += 1
        if eligible and state['seen'] == 17:
            from habana_frameworks.torch.utils.experimental import _data_ptr
            import torch.distributed as dist
            from vllm.distributed import get_tp_group
            hc.mark_step(); torch.hpu.synchronize()
            check(api.trace_start(int(_data_ptr(storage)), storage.numel()))
            state['started'] = True
            dist.barrier(group=get_tp_group().cpu_group)
        begin = ctypes.c_uint64()
        if state['started']:
            check(api.synProfilerGetCurrentTimeNS(ctypes.byref(begin)))
        result = original(*args, **kwargs)
        if state['started']:
            check(api.synProfilerAddCustomMeasurement(b'runner.sample_tokens', begin.value))
            state['samples'] += 1
            if state['samples'] == 3:
                # The synchronous runner has already copied sampled IDs to CPU.
                finish(state)
        return result

    runner.sample_tokens = sample
    runner._sdk_profile_state = state
    return dict(rank=worker.rank, status='ARMED', required_bytes=required.value)


def check(code):
    if code:
        raise RuntimeError(f'SDK profiler status {code}')


def finish(state):
    counts = (ctypes.c_uint64*2)()
    code = state['api'].trace_stop(str(state['path']).encode(), counts)
    state['started'] = False
    check(code)
    state['bytes'], state['entries'] = list(counts)
