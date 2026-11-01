"""Out-of-band probes for ordinary workers via vLLM's callable RPC API."""
from functools import partial

_METHODS = frozenset({
    'framework_step_graph', 'framework_compiled_regions', 'framework_input_layout',
    'framework_event_batching', 'framework_input_traversal', 'framework_execution_snapshot',
    'framework_timing', 'framework_submission_timing', 'framework_prequeue_probe',
    'framework_device_profile', 'production_quality_capture', 'host_placement_snapshot',
    'host_execution_policy',
})


def on_worker(worker, method, arguments, keyword_arguments):
    if method not in _METHODS:
        raise ValueError('Unknown validation operation: ' + method)
    from gaudi_kernels.serving.diagnostics.worker import DiagnosticHPUWorker
    if method == 'production_quality_capture' and not getattr(worker, '_quality_probe_installed', False):
        # Installed only when numerical checks begin, after measured requests.
        import vllm_gaudi.v1.worker.hpu_model_runner as hpu
        from gaudi_kernels.serving.executor.production_quality import install_runner
        install_runner(hpu, runner_cls=type(worker.model_runner))
        worker._quality_probe_installed = True
    return getattr(DiagnosticHPUWorker, method)(worker, *arguments, **keyword_arguments)


def call(llm, method, *, args=(), kwargs=None, production=False):
    if not production:
        return llm.collective_rpc(method, args=args, kwargs=kwargs)
    return llm.collective_rpc(partial(on_worker, method=method, arguments=args,
                                      keyword_arguments=kwargs or {}))
