"""Bounded diagnostic host timings; no tensor reads, synchronization or replay hooks."""
from functools import wraps
from time import perf_counter_ns


def configure(worker, enabled):
    if type(enabled) is not bool:
        raise ValueError('enabled must be bool')
    if hasattr(worker, '_framework_timing_state'):
        if enabled:
            raise ValueError('Disable the current diagnostic before starting another')
        state = worker._framework_timing_state
        for owner, name, original in state['originals']:
            setattr(owner, name, original)
        del worker._framework_timing_state
        return dict(rank=worker.rank, records=state['records'], dropped=state['dropped'],
                    scope='Nested host wall durations; device waits may occur inside calls; '
                          'do not sum parent and child durations or infer GPU compute time')
    if not enabled:
        return dict(rank=worker.rank, records=[], dropped=0)
    runner = worker.model_runner
    state = dict(originals=[], records=[], dropped=0)

    def attach(owner, name, metadata_position=None):
        if not hasattr(owner, name):
            return
        original = getattr(owner, name)
        label = ('model.' if owner is runner.model else 'runner.') + name

        @wraps(original)
        def measured(*args, **kwargs):
            meta = kwargs.get('attn_metadata')
            if meta is None and metadata_position is not None and len(args) > metadata_position:
                meta = args[metadata_position]
            prompt = getattr(meta, 'is_prompt', None)
            phase = ('prefill' if prompt else 'decode') if type(prompt) is bool else None
            begin = perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                end = perf_counter_ns()
                if len(state['records']) < 8192:
                    state['records'].append(dict(name=label, begin_ns=begin, end_ns=end,
                        duration_ns=end-begin, phase=phase))
                else:
                    state['dropped'] += 1
        state['originals'].append((owner, name, original))
        setattr(owner, name, measured)

    for name in ('execute_model', 'sample_tokens', '_prepare_inputs', '_prepare_sampling',
                 '_run_sampling', 'copy_input_to_device', 'copy_output_to_cpu'):
        attach(runner, name)
    attach(runner, '_execute_model_generic', 2)
    attach(runner.model, 'forward')
    attach(runner.model, 'compute_logits')
    worker._framework_timing_state = state
    return dict(rank=worker.rank, enabled=True, scope='Diagnostic only; original execution path retained')
