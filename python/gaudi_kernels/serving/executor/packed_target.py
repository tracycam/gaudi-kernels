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
    features: tuple = ()
    feature_layers: tuple = ()


class PackedTargetExecutor:
    def __init__(self, runner, *, kv_capacity=None, session=None, feature_layers=()):
        self.runner = runner
        self.adapter = runner.model
        self.model = self.adapter.model
        if (type(feature_layers) is not tuple or
                any(type(i) is not int or i < 0 for i in feature_layers) or
                tuple(sorted(set(feature_layers))) != feature_layers):
            raise ValueError('Feature layers require unique ascending decoder indices')
        self.feature_layers = feature_layers
        self.feature_holder = None
        if feature_layers:
            holders = [module for module in self.model.modules()
                       if callable(getattr(module, '_set_aux_hidden_state_layers', None))]
            if len(holders) != 1:
                raise ValueError('Target must expose exactly one auxiliary decoder feature interface')
            self.feature_holder = holders[0]
            if (getattr(self.feature_holder, 'start_layer', None) != 0 or
                    feature_layers[-1] >= getattr(self.feature_holder, 'end_layer', 0)):
                raise ValueError('Feature capture requires complete local decoder ownership')
        geometries = {}
        from gaudi_kernels.serving.packed_backend import PackedDiffKVImpl
        for name, layer in runner.vllm_config.compilation_config.static_forward_context.items():
            impl = getattr(layer, 'impl', None)
            if isinstance(impl, PackedDiffKVImpl):
                geometries[name] = (impl.num_kv_heads, impl.head_size, impl.head_size_v)
            elif impl is not None:
                raise ValueError('Loaded attention backend has no packed consumer: ' + name)
        self.session = session if session is not None else PackedKVSession(
            geometries, kv_capacity, device=runner.device, dtype=runner.vllm_config.model_config.dtype)

    @torch.inference_mode()
    def execute(self, batch):
        return self.execute_prepared(batch, self.session.prepare(batch))

    @torch.inference_mode()
    def execute_prepared(self, batch, metadata, inputs=None):
        """Consume an owned transaction and optionally persistent input buffers.

        Preparation/input transfer must finish before recording device commands.
        This boundary is experimental; it does not enable native serving.
        """
        from vllm.forward_context import set_forward_context
        import habana_frameworks.torch.core as htcore
        if self.session.pending is not metadata or metadata.batch is not batch:
            raise ValueError('Prepared target transaction is stale or belongs to another batch')
        old_feature_layers = None
        try:
            # Deliberately exclude capacity padding before embedding/projections.
            encoded = batch.encoded()
            if inputs is None:
                ids = torch.tensor(encoded['token_ids'][:batch.num_tokens], dtype=torch.int32, device=self.runner.device)
                positions = torch.tensor(encoded['positions'][:batch.num_tokens], dtype=torch.int64, device=self.runner.device)
                indices = torch.tensor(batch.logits_indices, dtype=torch.int64, device=self.runner.device)
            else:
                inputs.validate(batch)
                ids, positions, indices = inputs.ids, inputs.positions, inputs.logits_indices
            prepare_rope = self.adapter._rotary_prepare_cos_sin
            if prepare_rope is not None:
                prepare_rope(positions.reshape(1, -1), recompute_cos_sin=self.adapter.recompute_cos_sin)
            flat_positions = self.adapter.flatten_positions(positions)
            if self.feature_holder is not None:
                old_feature_layers = self.feature_holder.aux_hidden_state_layers
                # vLLM boundary i+1 is the residual-completed output of decoder
                # i, before the next layer and before final model normalization.
                self.feature_holder._set_aux_hidden_state_layers(tuple(i + 1 for i in self.feature_layers))
            with set_forward_context(metadata, self.runner.vllm_config, num_tokens=batch.num_tokens,
                                     skip_compiled=True):
                hidden = self.model(input_ids=ids, positions=flat_positions)
                features = ()
                if isinstance(hidden, tuple):
                    if self.feature_layers:
                        if len(hidden) != 2 or len(hidden[1]) != len(self.feature_layers):
                            raise ValueError('Target returned incomplete auxiliary features')
                        features = tuple(t.clone() for t in hidden[1])
                        if any(t.shape != hidden[0].shape for t in features):
                            raise ValueError('Auxiliary target feature geometry differs from hidden output')
                    hidden = hidden[0]
                elif self.feature_layers:
                    raise ValueError('Target did not return requested auxiliary features')
                # Framework/collective outputs can be reusable buffers. Keep
                # this result independent of the next forward, including one
                # from another executor sharing the loaded model.
                hidden = hidden.clone()
                logits = None
                if batch.logits_indices:
                    logits = self.model.compute_logits(hidden.index_select(0, indices)).clone()
            htcore.mark_step()
            torch.hpu.synchronize()
            return PackedTargetResult(hidden, logits, batch.logits_owners, metadata, features, self.feature_layers)
        except BaseException:
            # Device work must finish before the pool becomes reusable.
            htcore.mark_step()
            torch.hpu.synchronize()
            self.session.abort(metadata)
            raise
        finally:
            if old_feature_layers is not None:
                self.feature_holder._set_aux_hidden_state_layers(old_feature_layers)
            if self.adapter._rotary_prepare_cos_sin is not None:
                self.adapter._reset_rotary_cos_sin()

    def commit(self, result, prefix_lengths):
        # execute() is synchronizing for diagnostics. Native tickets are pending.
        self.session.commit(result.transaction, prefix_lengths)

    def abort(self, result):
        self.session.abort(result.transaction)
