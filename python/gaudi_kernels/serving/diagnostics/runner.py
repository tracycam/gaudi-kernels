"""Validation-only overrides around the ordinary serving runner."""
import vllm_gaudi.v1.worker.hpu_model_runner as hpu
from gaudi_kernels.serving.models.mimo_runner import MimoHPUModelRunner
from gaudi_kernels.serving.executor.production_quality import install_runner


class DiagnosticHPUModelRunner(MimoHPUModelRunner):
    def sample_tokens(self, grammar_output):
        if getattr(self, '_packed_scheduled_enabled', False):
            from gaudi_kernels.serving.executor.scheduled_target import sample_scheduled
            return sample_scheduled(self, grammar_output)
        return super().sample_tokens(grammar_output)

    def _prepare_inputs(self, scheduler_output, num_prefills, num_decodes, warmup=False):
        # Observe the scheduling source before the legacy plugin separates
        # prefill/decode and pads queries. No extra packing in timed replay.
        self._scheduled_token_batch = None
        transfers = getattr(self, '_input_audit', None)
        observing = (getattr(self, '_scheduled_token_audit', False) or
                     (transfers is not None and transfers.audit_inputs is not None))
        if not warmup and observing:
            from gaudi_kernels.serving.executor.scheduled_tokens import from_vllm
            self._scheduled_token_batch = from_vllm(self.input_batch, scheduler_output, self.requests)
            if getattr(self, '_scheduled_token_audit', False):
                records = self._scheduled_token_records
                if len(records) < 512:
                    batch = self._scheduled_token_batch
                    records.append({'request_ids': batch.request_ids, 'query_lengths': batch.query_lengths,
                                    'query_start_loc': batch.query_start_loc, 'valid_rows': batch.num_tokens,
                                    'query_kinds': tuple(r.kind for r in batch.requests),
                                    'logits_indices': batch.logits_indices})
                else:
                    self._scheduled_token_dropped += 1
        return super()._prepare_inputs(scheduler_output, num_prefills, num_decodes, warmup)

    def _create_decode_input_data(self, num_decodes, num_scheduled_tokens, context_lens,
                                  block_table_cpu_tensor, scheduler_output=None):
        result = super()._create_decode_input_data(num_decodes, num_scheduled_tokens,
                                                   context_lens, block_table_cpu_tensor, scheduler_output)
        self._native_query_layout=None
        transfers=getattr(self,'_input_audit',None)
        if transfers is not None and transfers.audit_inputs is not None and scheduler_output is not None:
            from gaudi_kernels.serving.executor.query_layout import DecodeQueries
            self._native_query_layout=DecodeQueries(
                tuple(self.input_batch.req_ids[:num_decodes]),
                tuple(int(v) for v in num_scheduled_tokens[:num_decodes]),
                tuple(int(v) for v in context_lens[:num_decodes]),result.token_ids.numel(),
                bool(scheduler_output.scheduled_spec_decode_tokens))
        return result

    def copy_input_to_device(self, role, source, **kwargs):
        audit = getattr(self, '_input_audit', None)
        if audit is not None:
            audit.capture(role, source)
        return super().copy_input_to_device(role, source, **kwargs)


install_runner(hpu, runner_cls=DiagnosticHPUModelRunner)
