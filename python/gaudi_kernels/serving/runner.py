"""Repository-owned runner selected through the vLLM worker factory."""
import vllm_gaudi.v1.worker.hpu_model_runner as hpu

from gaudi_kernels.serving.graph_options import GraphOptions


class FrameworkHPUModelRunner(hpu.HPUModelRunner):
    def speculative_row_budget(self):
        return 0

    def _create_speculative_proposer(self):
        if self.speculative_config.method == 'dflash':
            from gaudi_kernels.serving.draft.proposer import HpuDFlashProposer
            return HpuDFlashProposer(self.vllm_config, self.device, self,
                                    row_budget=self.speculative_row_budget())
        return super()._create_speculative_proposer()

    def graph_options(self):
        options = getattr(self, '_graph_options', None)
        if options is None:
            options = self._graph_options = GraphOptions.from_vllm(self.vllm_config)
        return options

    def initialize_kv_cache(self, *args, **kwargs):
        # Like model parameters, KV buffers are graph-owned module state.
        # Reinitialization is a lifecycle boundary, never a token-time check.
        if getattr(self, '_step_graph_programs', None):
            import torch
            import habana_frameworks.torch.core as hc
            hc.mark_step()
            torch.hpu.synchronize()
            self._step_graph_programs.clear()
        self._step_graph_sample = None
        self._step_graph_spec_sample = None
        self._prepared_decode_inputs = None
        self._prepared_spec_decode_inputs = None
        return super().initialize_kv_cache(*args, **kwargs)

    def _step_graph_enabled(self):
        override = getattr(self, '_step_graph_override', None)
        return (self.graph_options().scope == 'step'
                if override is None else override)

    def _execute_model_generic(self, token_ids, position_ids, attn_metadata, logits_indices,
                               kv_caches, lora_logits_mask, lora_mask, warmup_mode=False,
                               inputs_embeds=None, model_mm_kwargs=None):
        original = super()._execute_model_generic
        if self._step_graph_enabled():
            from gaudi_kernels.serving.decode_graph import execute
            return execute(self, original, token_ids, position_ids, attn_metadata, logits_indices,
                           kv_caches, lora_logits_mask, lora_mask, warmup_mode,
                           inputs_embeds, model_mm_kwargs)
        return original(token_ids, position_ids, attn_metadata, logits_indices, kv_caches,
                        lora_logits_mask, lora_mask, warmup_mode, inputs_embeds, model_mm_kwargs)

    def _run_sampling(self, batch_changed, logits_device, request_ids=None, pad_to=None, logits_requests=None):
        pending = self.__dict__.pop('_step_graph_sample', None)
        if pending is not None:
            if pending[0] is not logits_device:
                raise RuntimeError('Step graph sampling consumer does not match its logits producer')
            return pending[1], pending[2]
        return super()._run_sampling(batch_changed, logits_device, request_ids, pad_to, logits_requests)

    def _sample_decode(self, batch_changed, logits, request_ids, spec_metadata):
        if spec_metadata is not None and (
                getattr(self, '_step_graph_spec_sample', None) is not None or self._step_graph_enabled()):
            from gaudi_kernels.serving import decode_graph
            if hasattr(decode_graph, 'sample_speculative'):
                return decode_graph.sample_speculative(
                    self, super()._sample_decode, batch_changed, logits, request_ids, spec_metadata)
        return super()._sample_decode(batch_changed, logits, request_ids, spec_metadata)

    def _prepare_inputs(self, scheduler_output, num_prefills, num_decodes, warmup=False):
        self._step_decode_count = num_decodes if num_prefills == 0 and not warmup else 0
        if (self._step_decode_count and self._step_graph_enabled() and
                getattr(self, '_prepared_decode_inputs_enabled', self.graph_options().reuse_inputs)):
            from gaudi_kernels.serving.decode_inputs import prepare_single_query_values
            scheduled = prepare_single_query_values(self, scheduler_output, num_decodes)
            if scheduled is not None:
                return (self._prepare_prefill_inputs(0, num_decodes, scheduled),
                        self._prepare_decode_inputs(num_decodes, scheduled, scheduler_output))
        return super()._prepare_inputs(scheduler_output, num_prefills, num_decodes, warmup)

    def _prepare_spec_decode_inputs(self, scheduler_output, logits_indices, token_ids_device, max_num_sampled_tokens):
        if (getattr(self, '_step_decode_count', 0) and self._step_graph_enabled() and
                getattr(self, '_prepared_decode_inputs_enabled', self.graph_options().reuse_inputs)):
            from gaudi_kernels.serving.decode_inputs import prepare_spec_decode_inputs
            return prepare_spec_decode_inputs(
                self, super()._prepare_spec_decode_inputs, scheduler_output,
                logits_indices, token_ids_device, max_num_sampled_tokens)
        return super()._prepare_spec_decode_inputs(
            scheduler_output, logits_indices, token_ids_device, max_num_sampled_tokens)

    def _create_decode_input_data(self, num_decodes, num_scheduled_tokens, context_lens,
                                  block_table_cpu_tensor, scheduler_output=None):
        from gaudi_kernels.serving.decode_inputs import PreparedDecodeInputs, eligibility_reason
        enabled = getattr(self, '_prepared_decode_inputs_enabled', self.graph_options().reuse_inputs) and self._step_graph_enabled()
        reason = eligibility_reason(self, num_decodes, num_scheduled_tokens) if enabled else 'disabled'
        self._prepared_decode_input_rejection = reason
        prepared = reason is None
        cache = getattr(self, '_prepared_decode_inputs', None)
        result = (cache.update(self, num_decodes, context_lens, block_table_cpu_tensor,
                               num_scheduled_tokens, scheduler_output)
                  if prepared and cache is not None else None)
        if result is None:
            self._prepared_input_capture = {} if prepared else None
            try:
                result=super()._create_decode_input_data(num_decodes,num_scheduled_tokens,context_lens,
                                                        block_table_cpu_tensor,scheduler_output)
                self._prepared_decode_inputs = (PreparedDecodeInputs(
                    self, num_decodes, context_lens, block_table_cpu_tensor, result,
                    self._prepared_input_capture, num_scheduled_tokens, scheduler_output)
                    if prepared else None)
            finally:
                self._prepared_input_capture = None
        return result

    def copy_input_to_device(self, role, source, **kwargs):
        if (self._step_graph_enabled() and getattr(self, '_step_decode_count', 0) and
                self.graph_options().reuse_inputs and not self.use_async_scheduling):
            from habana_frameworks.torch.hpu import InputBufferCache
            device = kwargs.get('device', self.device)
            caches = self.__dict__.setdefault('_step_input_buffers', {})
            key = str(device)
            if key not in caches:
                caches[key] = InputBufferCache(device, max_entries=self.graph_options().max_input_buckets)
            result = caches[key].copy(role, source, non_blocking=True)
        else:
            result = super().copy_input_to_device(role, source, **kwargs)
        capture = getattr(self, '_prepared_input_capture', None)
        if capture is not None:
            capture[role] = (source, result)
        return result

    def copy_output_to_cpu(self, role, source):
        return super().copy_output_to_cpu(role, source)

    def _wrap_model_adapter(self, *args, **kwargs):
        adapter = hpu.HpuModelAdapter(*args, **kwargs)
        if hpu.htorch.utils.internal.is_lazy():
            return hpu.htorch.hpu.wrap_in_hpu_graph(adapter, disable_tensor_cache=False, asynchronous=False,
                                                   max_graphs=self.graph_options().max_cached_graphs)
        return adapter

    def _use_graphs(self, attn_metadata, batch_size):
        if getattr(self, '_inside_step_graph', False):
            return False
        return super()._use_graphs(attn_metadata, batch_size)

    def shutdown_inc(self):
        self._prepared_decode_inputs = None
        self._prepared_spec_decode_inputs = None
        if getattr(self, '_step_graph_programs', None) or getattr(self, '_step_input_buffers', None):
            import torch
            import habana_frameworks.torch.core as hc
            hc.mark_step()
            torch.hpu.synchronize()
            self._step_graph_sample = None
            self._step_graph_spec_sample = None
            getattr(self, '_step_graph_programs', {}).clear()
            getattr(self, '_step_input_buffers', {}).clear()
        shutdown = getattr(super(), 'shutdown_inc', None)
        if shutdown is not None:
            shutdown()
