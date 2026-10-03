"""Actual vLLM model consumer for compact target-only transactions.

Shares the loaded model and its precision/kernel policy. This eager diagnostic
entry point does not bypass the native serving admission gates or the scheduler.
"""
from dataclasses import dataclass

import torch

from gaudi_kernels.serving.executor.packed_attention import PackedKVSession


@dataclass
class PackedTargetResult:
    hidden: torch.Tensor
    logits: torch.Tensor | None
    output_owners: tuple
    transaction: object


class PackedTargetExecutor:
    def __init__(self, runner, *, kv_capacity):
        self.runner = runner
        self.adapter = runner.model
        self.model = self.adapter.model
        geometries = {}
        from gaudi_kernels.serving.packed_backend import PackedDiffKVImpl
        for name, layer in runner.vllm_config.compilation_config.static_forward_context.items():
            impl = getattr(layer, 'impl', None)
            if isinstance(impl, PackedDiffKVImpl):
                geometries[name] = (impl.num_kv_heads, impl.head_size, impl.head_size_v)
            elif impl is not None:
                raise ValueError('Loaded attention backend has no packed consumer: ' + name)
        self.session = PackedKVSession(geometries, kv_capacity, device=runner.device,
                                       dtype=runner.vllm_config.model_config.dtype)

    @torch.inference_mode()
    def execute(self, batch):
        from vllm.forward_context import set_forward_context
        import habana_frameworks.torch.core as htcore
        metadata = self.session.prepare(batch)
        try:
            # Deliberately exclude capacity padding before embedding/projections.
            encoded = batch.encoded()
            ids = torch.tensor(encoded['token_ids'][:batch.num_tokens], dtype=torch.int32, device=self.runner.device)
            positions = torch.tensor(encoded['positions'][:batch.num_tokens], dtype=torch.int64, device=self.runner.device)
            prepare_rope = self.adapter._rotary_prepare_cos_sin
            if prepare_rope is not None:
                prepare_rope(positions.reshape(1, -1), recompute_cos_sin=self.adapter.recompute_cos_sin)
            flat_positions = self.adapter.flatten_positions(positions)
            with set_forward_context(metadata, self.runner.vllm_config, num_tokens=batch.num_tokens,
                                     skip_compiled=True):
                hidden = self.model(input_ids=ids, positions=flat_positions)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                logits = None
                if batch.logits_indices:
                    indices = torch.tensor(batch.logits_indices, dtype=torch.int64, device=self.runner.device)
                    logits = self.model.compute_logits(hidden.index_select(0, indices))
            htcore.mark_step()
            torch.hpu.synchronize()
            return PackedTargetResult(hidden, logits, batch.logits_owners, metadata)
        except BaseException:
            # Device work must finish before the pool becomes reusable.
            htcore.mark_step()
            torch.hpu.synchronize()
            self.session.abort(metadata)
            raise
        finally:
            if self.adapter._rotary_prepare_cos_sin is not None:
                self.adapter._reset_rotary_cos_sin()

    def commit(self, result, prefix_lengths):
        # execute() is synchronizing for diagnostics. Native tickets are pending.
        self.session.commit(result.transaction, prefix_lengths)

    def abort(self, result):
        self.session.abort(result.transaction)
