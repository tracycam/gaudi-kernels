"""Bounded public API timing after performance requests, with no synchronization."""
import ctypes
from time import perf_counter_ns

LABELS=('launch', 'launch_ext', 'all_gather', 'all_reduce', 'event_record',
        'stream_wait', 'event_sync')


def configure(worker, enabled):
    import torch
    import habana_frameworks.torch.core as hc
    runner=worker.model_runner
    hc.mark_step(); torch.hpu.synchronize()
    if not enabled:
        state=runner.__dict__.pop('_submission_timing', None)
        if state is None:
            return dict(rank=worker.rank, rows=[])
        runner._execute_model_generic=state['original']
        return dict(rank=worker.rank, rows=state['rows'],
                    scope='Inclusive CPU API duration; event_sync can run concurrently on GC thread; do not sum it into submit time')
    if hasattr(runner, '_submission_timing'):
        raise ValueError('Submission observer already armed')
    api=ctypes.CDLL(None)
    api.api_timing_end.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    original=runner._execute_model_generic
    state=dict(original=original, rows=[], seen=0)
    def execute(*args, **kwargs):
        metadata=args[2] if len(args)>2 else kwargs['attn_metadata']
        if metadata.is_prompt:
            return original(*args, **kwargs)
        state['seen']+=1
        if not 17 <= state['seen'] <= 48:
            return original(*args, **kwargs)
        if api.api_timing_begin():
            raise RuntimeError('Overlapping diagnostic windows')
        before=perf_counter_ns()
        try:
            return original(*args, **kwargs)
        finally:
            wall=perf_counter_ns()-before
            values=(ctypes.c_uint64*(4*len(LABELS)))()
            api.api_timing_end(values)
            state['rows'].append(dict(wall_ms=wall/1e6, apis={name:dict(
                count=values[4*i], inclusive_ms=values[4*i+1]/1e6,
                outer_count=values[4*i+2], outer_ms=values[4*i+3]/1e6)
                for i,name in enumerate(LABELS)}))
    runner._execute_model_generic=execute
    runner._submission_timing=state
    return dict(rank=worker.rank, status='ARMED')
