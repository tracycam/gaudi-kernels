"""vLLM worker extension: explicit runner factory and public utility methods."""
from vllm_gaudi.v1.worker.hpu_worker import HPUWorker

from gaudi_kernels.serving.executor.native_backend import configure, summary, configure_profile
from gaudi_kernels.serving.executor.production_quality import configure_capture, configure_policy, snapshot
from gaudi_kernels.serving.host_placement import snapshot as host_snapshot


class NativeHPUWorker(HPUWorker):
    def load_model(self, *args, **kwargs):
        from gaudi_kernels.serving.packed_backend import register
        register()
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

    def packed_target_probe(self, plan):
        from gaudi_kernels.serving.diagnostics.packed_model import on_worker
        return on_worker(self, plan)

    def packed_scheduled_configure(self, enabled):
        from gaudi_kernels.engine.context import context
        runner = self.model_runner
        parallel = runner.vllm_config.parallel_config
        if type(enabled) is not bool or runner.input_batch.num_reqs or context().native.active:
            raise ValueError('Packed scheduling requires explicit bool, no requests and native replay disabled')
        if (runner.speculative_config is not None or runner.use_async_scheduling or
                parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1 or
                runner.vllm_config.lora_config is not None or runner.vllm_config.kv_transfer_config is not None):
            raise ValueError('Packed scheduled probe requires synchronous plain TP target execution')
        records = getattr(runner, '_packed_scheduled_records', [])
        if enabled:
            if getattr(runner, '_packed_scheduled_failed', False):
                raise ValueError('Failed packed step requires a fresh worker')
            runner._packed_scheduled_records = []
        runner._packed_scheduled_enabled = enabled
        return {'rank': self.rank, 'enabled': enabled, 'records': records,
                'scope': 'experimental actual allocator/sampler execution; synchronizing diagnostics, no native TPS claim'}

    def scheduled_token_audit(self, enabled=None, clear=False):
        """Bounded source-level diagnostics; no tensor values or device access."""
        if enabled is not None and type(enabled) is not bool:
            raise ValueError('Token audit enabled must be a bool or None')
        if type(clear) is not bool:
            raise ValueError('Token audit clear must be a bool')
        runner = self.model_runner
        if clear or not hasattr(runner, '_scheduled_token_records'):
            runner._scheduled_token_records = []
            runner._scheduled_token_dropped = 0
        if enabled is not None:
            runner._scheduled_token_audit = enabled
        return {'rank': self.rank, 'enabled': getattr(runner, '_scheduled_token_audit', False),
                'records': list(runner._scheduled_token_records), 'dropped': runner._scheduled_token_dropped,
                'scope': 'packed scheduling source; legacy model execution is unchanged'}

    native_configure = configure
    native_summary = summary
    native_profile = configure_profile
    production_quality_capture = configure_capture
    production_policy = configure_policy
    production_snapshot = snapshot
    host_placement_snapshot = host_snapshot
