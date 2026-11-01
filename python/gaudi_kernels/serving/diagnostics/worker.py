"""Explicit validation worker; diagnostics never install on production workers."""
from gaudi_kernels.serving.worker import MimoHPUWorker
from gaudi_kernels.serving.executor.production_quality import configure_capture, snapshot
from gaudi_kernels.serving.policy import configure_policy
from gaudi_kernels.serving.host_placement import snapshot as host_snapshot


class DiagnosticHPUWorker(MimoHPUWorker):
    def get_model_runner_cls(self):
        from gaudi_kernels.serving.diagnostics.runner import DiagnosticHPUModelRunner
        return DiagnosticHPUModelRunner

    def packed_target_probe(self, plan):
        from gaudi_kernels.serving.diagnostics.packed_model import on_worker
        return on_worker(self, plan)

    def packed_feature_probe(self, feature_layers, native_swa=False):
        from gaudi_kernels.serving.diagnostics.packed_features import on_worker
        return on_worker(self, feature_layers, native_swa)

    def dflash_cycle_probe(self, plan):
        from gaudi_kernels.serving.diagnostics.dflash_cycle import on_worker
        return on_worker(self, plan)

    def moe_rows_probe(self, plan):
        from gaudi_kernels.serving.diagnostics.moe_rows import on_worker
        return on_worker(self, plan)

    def packed_scheduled_configure(self, enabled):
        from gaudi_kernels.engine.context import context
        runner = self.model_runner
        parallel = runner.vllm_config.parallel_config
        if type(enabled) is not bool or runner.input_batch.num_reqs:
            raise ValueError('Packed scheduling requires explicit bool and no requests')
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

    def framework_execution_snapshot(self):
        from gaudi_kernels.serving.diagnostics.bridge import snapshot
        return snapshot(self)

    def framework_event_batching(self, enabled):
        from gaudi_kernels.serving.diagnostics.bridge import event_batching
        return event_batching(self, enabled)

    def framework_compiled_regions(self, enabled):
        from gaudi_kernels.serving.diagnostics.bridge import event_batching
        return event_batching(self, enabled, compiled=True)

    def framework_input_traversal(self, enabled):
        from gaudi_kernels.serving.diagnostics.bridge import input_traversal
        return input_traversal(self, enabled)

    def framework_step_graph(self, enabled):
        from gaudi_kernels.serving.decode_graph import configure
        return configure(self, enabled)

    def framework_input_layout(self, enabled):
        from gaudi_kernels.serving.decode_inputs import configure
        return configure(self, enabled)

    def framework_prequeue_probe(self, library=None):
        from gaudi_kernels.serving.diagnostics.prequeue import configure
        return configure(self, library)

    def framework_submission_timing(self, enabled):
        from gaudi_kernels.serving.diagnostics.submission_timing import configure
        return configure(self, enabled)

    def framework_device_profile(self, library, tag, enabled=True):
        from gaudi_kernels.serving.diagnostics.device_profile import configure
        return configure(self, library, tag, enabled)

    def framework_timing(self, enabled, cpu_trace=False, device_trace=False):
        from gaudi_kernels.serving.diagnostics.bridge_timing import configure
        return configure(self, enabled, cpu_trace=cpu_trace, device_trace=device_trace)

    def host_execution_policy(self, mode=None, traces=None):
        from gaudi_kernels.serving.diagnostics.host_policy import configure
        return configure(self, mode, traces)

    def device_memory_snapshot(self, reset_peak=False):
        """Out-of-band allocator telemetry; never called inside inference."""
        import torch
        import habana_frameworks.torch.core as hc
        if type(reset_peak) is not bool:
            raise ValueError('reset_peak must be bool')
        hc.mark_step(); torch.hpu.synchronize()
        if reset_peak:
            torch.hpu.reset_peak_memory_stats()
        return {'rank':self.rank,'allocated':torch.hpu.memory_allocated(),
                'reserved':torch.hpu.memory_reserved(),
                'max_allocated':torch.hpu.max_memory_allocated(),
                'max_reserved':torch.hpu.max_memory_reserved(),
                'scope':'Torch allocator; excludes driver/firmware allocations'}

    production_quality_capture = configure_capture
    production_policy = configure_policy
    production_snapshot = snapshot
    host_placement_snapshot = host_snapshot
