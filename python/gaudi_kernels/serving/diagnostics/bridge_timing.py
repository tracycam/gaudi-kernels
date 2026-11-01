"""Bounded diagnostic host timings; no tensor reads, synchronization or replay hooks."""
from functools import wraps
from time import perf_counter_ns


def restore_methods(state):
    for owner, name, existed, value in reversed(state['originals']):
        if existed:
            setattr(owner, name, value)
        else:
            delattr(owner, name)


def configure(worker, enabled, *, cpu_trace=False, device_trace=False):
    if any(type(value) is not bool for value in (enabled, cpu_trace, device_trace)):
        raise ValueError('enabled/cpu_trace/device_trace must be bool')
    if hasattr(worker, '_framework_timing_state'):
        if enabled:
            raise ValueError('Disable the current diagnostic before starting another')
        state = worker._framework_timing_state
        restore_methods(state)
        del worker._framework_timing_state
        trace_path = None
        if state.get('profiler_started'):
            state['profiler'].__exit__(None, None, None)
            from gaudi_kernels.engine.context import context
            prefix = 'device-trace' if state['device_trace'] else 'cpu-trace'
            trace_path = context().startup.run_dir / f'{prefix}-rank{worker.rank}.json'
            state['profiler'].export_chrome_trace(str(trace_path))
        if state['device_trace'] and not state['profiler_started']:
            raise RuntimeError('Requested device trace saw no decode metadata; observer restored')
        return dict(rank=worker.rank, records=state['records'], dropped=state['dropped'],
                    cpu_trace=str(trace_path) if trace_path and not state['device_trace'] else None,
                    device_trace=str(trace_path) if trace_path and state['device_trace'] else None,
                    hardware_trace=state['device_trace'],
                    scope='Nested host wall durations; device waits may occur inside calls; '
                          'do not sum parent and child durations or infer GPU compute time')
    if not enabled:
        return dict(rank=worker.rank, records=[], dropped=0)
    runner = worker.model_runner
    state = dict(originals=[], records=[], dropped=0, device_trace=device_trace, profiler_started=False)
    if cpu_trace or device_trace:
        import torch
        from habana_frameworks.torch.activity_profiler.activity_profiler import DebugActivity
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device_trace:
            activities.append(torch.profiler.ProfilerActivity.HPU)
        state['profiler'] = torch.profiler.profile(activities=activities,
                debug_activities=[DebugActivity.BRIDGE_FUNCTION_CALLS] if cpu_trace else None)
        if not device_trace:
            state['profiler'].__enter__()
            state['profiler_started'] = True

    def attach(owner, name, metadata_position=None, prefix=None):
        if not hasattr(owner, name):
            return
        original = getattr(owner, name)
        label = (prefix or ('model.' if owner is runner.model else 'runner.')) + name

        @wraps(original)
        def measured(*args, **kwargs):
            meta = kwargs.get('attn_metadata')
            if meta is None and metadata_position is not None and len(args) > metadata_position:
                meta = args[metadata_position]
            prompt = getattr(meta, 'is_prompt', None)
            phase = ('prefill' if prompt else 'decode') if type(prompt) is bool else None
            if state['device_trace'] and not state['profiler_started'] and phase == 'decode':
                # Capture only steady decode, not checkpoint load/capture/prefill.
                state['profiler'].__enter__()
                state['profiler_started'] = True
            begin = perf_counter_ns()
            try:
                if state['device_trace'] and state['profiler_started']:
                    import torch
                    with torch.profiler.record_function(label):
                        return original(*args, **kwargs)
                return original(*args, **kwargs)
            finally:
                end = perf_counter_ns()
                if len(state['records']) < 8192:
                    state['records'].append(dict(name=label, begin_ns=begin, end_ns=end,
                        duration_ns=end-begin, phase=phase))
                else:
                    state['dropped'] += 1
        # Restoring an inherited bound method as an instance attribute would
        # shadow later class hooks, including the independent quality probe.
        existing = vars(owner)
        state['originals'].append((owner, name, name in existing, existing.get(name)))
        setattr(owner, name, measured)

    for name in ('execute_model', 'sample_tokens', '_prepare_inputs', '_prepare_sampling',
                 '_run_sampling', 'copy_input_to_device', 'copy_output_to_cpu'):
        attach(runner, name)
    attach(runner, '_execute_model_generic', 2)
    attach(runner.model, 'forward')
    attach(runner.model, 'compute_logits')
    # One scope per public graph replay, never per tensor or recursive leaf.
    # This separates Python input handling from C++ replay without an SDK trace.
    from habana_frameworks.torch.hpu import graphs
    graph_classes = {graphs.HPUGraph}
    for fn in getattr(worker, '_input_traversal_ab', ()):
        graph_classes.add(fn.__globals__['HPUGraph'])
    for graph_class in graph_classes:
        attach(graph_class, 'replay', prefix='graph.')
    worker._framework_timing_state = state
    return dict(rank=worker.rank, enabled=True, scope='Diagnostic only; original execution path retained')
