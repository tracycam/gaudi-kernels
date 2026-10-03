"""Repository-owned runner selected through the vLLM worker factory."""
import vllm_gaudi.v1.worker.hpu_model_runner as hpu

from gaudi_kernels.engine.context import context
from gaudi_kernels.serving.model_adapter import adapter_class


class NativeHPUModelRunner(hpu.HPUModelRunner):
    def _prepare_inputs(self, scheduler_output, num_prefills, num_decodes, warmup=False):
        # Observe the scheduling source before the legacy plugin separates
        # prefill/decode and pads queries. No extra packing in timed replay.
        self._scheduled_token_batch = None
        transfers = getattr(self, '_native_transfers', None)
        if not warmup and transfers is not None and transfers.audit_inputs is not None:
            from gaudi_kernels.serving.executor.scheduled_tokens import from_vllm
            self._scheduled_token_batch = from_vllm(self.input_batch, scheduler_output, self.requests)
        return super()._prepare_inputs(scheduler_output, num_prefills, num_decodes, warmup)

    def _create_decode_input_data(self, num_decodes, num_scheduled_tokens, context_lens,
                                  block_table_cpu_tensor, scheduler_output=None):
        result=super()._create_decode_input_data(num_decodes,num_scheduled_tokens,context_lens,
                                                block_table_cpu_tensor,scheduler_output)
        self._native_query_layout=None
        transfers=getattr(self,'_native_transfers',None)
        if transfers is not None and transfers.audit_inputs is not None and scheduler_output is not None:
            from gaudi_kernels.serving.executor.query_layout import DecodeQueries
            self._native_query_layout=DecodeQueries(
                tuple(self.input_batch.req_ids[:num_decodes]),
                tuple(int(v) for v in num_scheduled_tokens[:num_decodes]),
                tuple(int(v) for v in context_lens[:num_decodes]),result.token_ids.numel(),
                bool(scheduler_output.scheduled_spec_decode_tokens))
        return result

    def copy_input_to_device(self, role, source, **kwargs):
        return self._native_transfers.upload(role, source, super().copy_input_to_device, **kwargs)

    def copy_output_to_cpu(self, role, source):
        return self._native_transfers.download(role, source, super().copy_output_to_cpu)

    def _wrap_model_adapter(self, *args, **kwargs):
        cls = adapter_class(hpu.HpuModelAdapter) if context().has('qkv_post') else hpu.HpuModelAdapter
        adapter = cls(*args, **kwargs)
        if hpu.htorch.utils.internal.is_lazy():
            return hpu.htorch.hpu.wrap_in_hpu_graph(adapter, disable_tensor_cache=False, asynchronous=False)
        return adapter

    def _use_graphs(self, attn_metadata, batch_size):
        if attn_metadata is not None and attn_metadata.is_prompt:
            return False
        return super()._use_graphs(attn_metadata, batch_size)

    def shutdown_inc(self):
        if getattr(self, '_native_backend', {}).get('plan') is not None:
            self._native_reset()
        shutdown = getattr(super(), 'shutdown_inc', None)
        if shutdown is not None:
            shutdown()


from gaudi_kernels.serving.executor.native_backend import install
from gaudi_kernels.serving.executor.production_quality import install_runner

install(hpu, runner_cls=NativeHPUModelRunner)
install_runner(hpu, runner_cls=NativeHPUModelRunner)
