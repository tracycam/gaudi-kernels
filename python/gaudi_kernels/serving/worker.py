"""vLLM worker extension: explicit runner factory and public utility methods."""
from vllm_gaudi.v1.worker.hpu_worker import HPUWorker

from gaudi_kernels.serving.executor.native_backend import configure, summary, configure_profile
from gaudi_kernels.serving.executor.production_quality import configure_capture, configure_policy, snapshot
from gaudi_kernels.serving.host_placement import snapshot as host_snapshot


class NativeHPUWorker(HPUWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        from gaudi_kernels.engine.context import context
        configure_policy(self, context().selection.to_dict())
        if context().selection.engine.runtime.native_enabled:
            runner=self.model_runner
            if runner.use_async_scheduling or runner.vllm_config.max_concurrent_batches!=1:
                raise ValueError('Native serving requires synchronous scheduling and one concurrent batch')
            if runner.speculative_config is not None:
                raise ValueError('Native speculative transaction is not admitted yet')
            configure(self,True,True,True,'compact')
        return result

    def get_model_runner_cls(self):
        from gaudi_kernels.serving.runner import NativeHPUModelRunner
        return NativeHPUModelRunner

    native_configure = configure
    native_summary = summary
    native_profile = configure_profile
    production_quality_capture = configure_capture
    production_policy = configure_policy
    production_snapshot = snapshot
    host_placement_snapshot = host_snapshot
