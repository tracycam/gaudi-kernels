# SPDX-License-Identifier: Apache-2.0
"""Target/drafter composition with device token edges and one delivery boundary.

The eager target currently synchronizes; this is a functional composition, not
a native serving plan. Recorder integration must own all storage/completion
leases. The target still owns its KV transaction and vLLM owns serving state.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from gaudi_kernels.engine.token_batch import QueryBatch, RequestResult, StepResult
from gaudi_kernels.serving.executor.greedy_verify import verify_logits
from gaudi_kernels.serving.executor.packed_bindings import PackedInputBuffers
from .context import CommittedContext


@dataclass
class DFlashCycleResult:
    target: object
    verification: object
    proposed_ids: torch.Tensor
    next_anchor_ids: torch.Tensor


class DFlashCycle:
    def __init__(self, target, draft, request_ids):
        self.target, self.draft = target, draft
        self.request_ids = tuple(request_ids)
        spec = draft.spec
        if (not self.request_ids or len(set(self.request_ids)) != len(self.request_ids) or
                tuple(target.feature_layers) != spec.target_layers):
            raise ValueError('Drafter request ownership and actual target feature layers must match')
        self.context = CommittedContext(layers=spec.layers, batch=len(self.request_ids), window=spec.window,
            kv_heads=spec.kv_heads, dim=spec.dim, device=target.runner.device)
        self.pending = None
        self.failed = False

    def append_features(self, target_result, committed_counts):
        batch = target_result.transaction.batch
        spec = self.draft.spec
        if (any(rid not in self.request_ids for rid in batch.request_ids) or target_result.feature_layers != spec.target_layers or
                len(target_result.features) != len(spec.target_layers) or
                committed_counts.shape != (len(self.request_ids),) or max(batch.query_lengths) > spec.window):
            raise ValueError('Context append requires owned bounded target features/counts')
        # Project valid packed rows once. Request padding happens only after
        # projection, for the context-ring writes; it cannot amplify FC reads.
        features = torch.cat(target_result.features, -1).unsqueeze(0)
        positions = torch.cat(target_result.transaction.query_positions)
        projected = self.draft.project_context(features, positions.unsqueeze(0))
        width = max(batch.query_lengths)
        spans = {rid: (begin, end) for rid, begin, end in
                 zip(batch.request_ids, batch.query_start_loc, batch.query_start_loc[1:])}
        rows = []
        for rid in self.request_ids:
            if rid in spans:
                begin, end = spans[rid]
                rows.append(F.pad(positions[begin:end], (0, width-(end-begin)), value=-1))
            else:
                rows.append(torch.full((width,), -1, device=positions.device, dtype=positions.dtype))
        ring_positions = torch.stack(rows)
        pairs = []
        for pair in projected:
            values = []
            for source in pair:
                segments = []
                for rid in self.request_ids:
                    if rid in spans:
                        begin, end = spans[rid]
                        segments.append(F.pad(source[0, begin:end], (0, 0, 0, 0, 0, width-(end-begin))))
                    else:
                        segments.append(source.new_zeros((width, *source.shape[2:])))
                values.append(torch.stack(segments))
            pairs.append(tuple(values))
        self.context.commit(tuple(pairs), ring_positions, committed_counts)

    @torch.inference_mode()
    def run_prepared(self, schedule, metadata, anchor_ids, remaining, *, eos_ids=(), inputs=None):
        if (self.failed or self.pending is not None or type(schedule) is not QueryBatch or
                schedule.request_ids != self.request_ids or self.target.session.pending is not metadata or
                metadata.batch is not schedule or any(r.kind not in ('decode', 'verify') for r in schedule.requests) or
                max(schedule.query_lengths) > self.draft.spec.block or
                schedule.logits_indices != tuple(range(schedule.num_tokens)) or
                anchor_ids.shape != (len(self.request_ids),) or remaining.shape != anchor_ids.shape or
                anchor_ids.device != self.context.positions.device or remaining.device != anchor_ids.device or
                any(t.dtype not in (torch.int32, torch.int64) for t in (anchor_ids, remaining))):
            raise ValueError('Invalid or unfinished DFlash/target cycle ownership/extent')
        try:
            binding = PackedInputBuffers(schedule, self.target.runner.device) if inputs is None else inputs
            starts = torch.stack([p[0] for p in metadata.query_positions])
            positions = starts[:, None] + torch.arange(self.draft.spec.block, device=starts.device)
            embedding = self.target.model.embed_input_ids(anchor_ids)
            context, context_positions, context_valid = self.context.state()
            hidden = self.draft(self.draft.noise(embedding), positions, context, context_positions, context_valid)
            logits = self.target.model.compute_logits(hidden.reshape(-1, self.draft.spec.hidden))
            proposals = logits.reshape(len(self.request_ids), self.draft.spec.block, -1).argmax(-1).to(torch.int32)
            # Query i consumes the anchor or preceding draft, not the target's
            # own predicted token at i. No device IDs are read back on CPU.
            queries = torch.cat((anchor_ids[:, None], proposals[:, :-1]), 1)
            packed_ids = torch.cat([queries[i, :n] for i, n in enumerate(schedule.query_lengths)])
            binding.update_device(schedule, packed_ids, torch.cat(metadata.query_positions))
            target = self.target.execute_prepared(schedule, metadata, binding)
            decision = verify_logits(target.logits, binding.ids, schedule.query_lengths, remaining, eos_ids=eos_ids)
            # Only accepted target features reach committed drafter context.
            self.append_features(target, decision.committed_input_counts)
            last = (decision.emitted_counts - 1).clamp(min=0).to(torch.int64)
            next_anchor = decision.emitted_ids.gather(1, last[:, None]).squeeze(1)
            next_anchor = torch.where(decision.emitted_counts > 0, next_anchor, anchor_ids)
            self.pending = DFlashCycleResult(target, decision, proposals, next_anchor)
            return self.pending
        except BaseException:
            # Context/cache writes may have started. Reusing this coordinator
            # requires fresh prefill, not a silent fallback after partial work.
            self.failed = True
            raise

    def finish_at_output_boundary(self, result):
        """Apply the delivered device-selected prefixes after device completion.

        This eager implementation drains at the output boundary. A native
        runtime must replace that drain with its completed ticket. Counts are
        read from the actual verifier; callers cannot substitute another prefix.
        """
        if self.failed or result is not self.pending:
            raise ValueError('Stale/failed DFlash cycle')
        try:
            if result.target.hidden.device.type == 'hpu':
                import habana_frameworks.torch.core as htcore
                htcore.mark_step()
                torch.hpu.synchronize()
            if not bool(result.verification.valid.cpu().all()):
                raise ValueError('Device verifier rejected an invalid budget')
            counts = tuple(result.verification.committed_input_counts.cpu().tolist())
            emitted = result.verification.emitted_ids.cpu().tolist()
            output = StepResult(tuple(RequestResult(r.request_id, r.query_length, n, tuple(row[:n]))
                                      for r, n, row in zip(result.target.transaction.batch.requests, counts, emitted)))
            self.target.commit(result.target, counts)
        except BaseException:
            self.failed = True
            raise
        self.pending = None
        return output
