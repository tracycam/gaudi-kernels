"""Explicit read/timing RPCs for a loopback validation server, never serving boot."""
from functools import wraps
import statistics
import time


class ApiWorkerExtension:
    def framework_target_logits(self, tag=None, max_calls=16):
        """Store complete, pre-sampler logits with exact teacher input identity.

        Readback is diagnostic only. Disabling restores the original method;
        neither the model nor its numerical/graph policy is replaced.
        """
        if tag is not None and (type(tag) is not str or not tag or
                                any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in tag)):
            raise ValueError('Bounded capture tag required')
        if type(max_calls) is not int or not 1 <= max_calls <= 128:
            raise ValueError('Capture requires 1..128 calls')
        if tag is None:
            previous = getattr(self, '_target_logits_original', None)
            if previous is not None:
                self.model_runner._execute_model_generic = previous
                self._target_logits_original = None
            return dict(rank=self.rank, frames=getattr(self, '_target_logits_frames', []))
        if getattr(self, '_target_logits_original', None) is not None:
            raise ValueError('Logit capture already active')
        from pathlib import Path
        from gaudi_kernels.engine.context import context
        destination = context().startup.run_dir / 'target-logits' / tag / f'rank{self.rank}'
        destination.mkdir(parents=True, exist_ok=False)
        runner = self.model_runner
        original = self._target_logits_original = runner._execute_model_generic
        self._target_logits_frames = []

        @wraps(original)
        def capture(*args, **kwargs):
            output = original(*args, **kwargs)
            if (self.rank != 0 or len(self._target_logits_frames) >= max_calls or
                    output[-1] is None or output[-1].numel() == 0):
                return output
            import torch
            import habana_frameworks.torch.core as hc
            hc.mark_step()
            data = dict(input_ids=args[0].detach().cpu(), positions=args[1].detach().cpu(),
                        logits_indices=args[3].detach().cpu(), is_prompt=args[2].is_prompt,
                        logits=output[-1].detach().cpu().float())
            path = destination / f'{len(self._target_logits_frames)}.pt'
            torch.save(data, path)
            self._target_logits_frames.append(dict(path=str(path), is_prompt=data['is_prompt'],
                input_ids=data['input_ids'].reshape(-1).tolist(), positions=data['positions'].reshape(-1).tolist(),
                logits_shape=list(data['logits'].shape)))
            return output

        runner._execute_model_generic = capture
        return dict(rank=self.rank, enabled=True, scope='Validation-only complete logits; never timed')

    def framework_grouped_prefill(self, enabled, min_mme_rows=1, slot_budget=0, padded_row_budget=0):
        if type(enabled) is not bool:
            raise ValueError('Explicit bool required')
        from gaudi_kernels.engine.context import context
        from gaudi_kernels.engine.selection import PolicySelection
        from gaudi_kernels.serving.policy import configure_policy
        document = context().selection.to_dict()
        dispatch = document['engine']['decode']['moe']['dispatch']
        dispatch.update(grouped_rows=[512] if enabled else [],
                        grouped_min_mme_rows=min_mme_rows, grouped_slot_budget=slot_budget,
                        grouped_padded_row_budget=padded_row_budget)
        # Validate before touching device state, then use the same drain/cache
        # retirement/installation path as production. Never mutate a selector
        # underneath an existing capture.
        PolicySelection.from_dict(document)
        changed = configure_policy(self, document)
        return dict(rank=self.rank, enabled=enabled, dispatch=changed['moe_dispatch'],
                    generation=self.model_runner._policy_generation,
                    scope='Validation dispatch via production policy lifecycle')

    def framework_execution_snapshot(self):
        from gaudi_kernels.serving.diagnostics.bridge import snapshot
        return snapshot(self)

    def framework_host_policy(self, mode=None):
        from gaudi_kernels.serving.diagnostics.host_policy import configure
        return configure(self, mode)

    def framework_phase_timing(self, enabled, synchronize=False):
        if enabled not in ('on', 'off'):
            raise ValueError('Phase timing requires on/off')
        if type(synchronize) is not bool:
            raise ValueError('synchronize must be bool')
        if enabled == 'on':
            if getattr(self, '_api_phase_wrappers', None):
                raise ValueError('Phase timing already active')
            self._api_phase_times = {}
            self._api_phase_events = []
            self._api_phase_wrappers = []

            def sync():
                if synchronize:
                    import torch
                    import habana_frameworks.torch.core as hc
                    hc.mark_step()
                    torch.hpu.synchronize()

            def install(owner, name, phase):
                original = getattr(owner, name)

                @wraps(original)
                def measured(*args, **kwargs):
                    sync()
                    begin = time.perf_counter_ns()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        sync()
                        end = time.perf_counter_ns()
                        rows = self._api_phase_times.setdefault(phase, [])
                        if len(rows) < 8192:
                            rows.append((end - begin) / 1e6)
                            event = dict(phase=phase, begin_ns=begin, end_ns=end, milliseconds=rows[-1])
                            if phase == 'target_forward':
                                tokens = args[0] if args else kwargs.get('token_ids')
                                metadata = args[2] if len(args) > 2 else kwargs.get('attn_metadata')
                                event.update(token_rows=tokens.numel() if tokens is not None else None,
                                             is_prompt=getattr(metadata, 'is_prompt', None))
                                lengths = getattr(metadata, 'seq_lens', None)
                                if isinstance(lengths, (list, tuple)):
                                    event['sequence_lengths'] = list(lengths)
                            self._api_phase_events.append(event)

                setattr(owner, name, measured)
                self._api_phase_wrappers.append((owner, name, original))

            runner = self.model_runner
            original_execute = runner.execute_model

            @wraps(original_execute)
            def scheduled(scheduler_output, *args, **kwargs):
                if len(self._api_phase_events) < 8192:
                    self._api_phase_events.append(dict(
                        phase='scheduler_input', begin_ns=time.perf_counter_ns(),
                        scheduled_tokens=dict(scheduler_output.num_scheduled_tokens),
                        draft_lengths={key: len(value) for key, value in
                                       scheduler_output.scheduled_spec_decode_tokens.items()}))
                return original_execute(scheduler_output, *args, **kwargs)

            runner.execute_model = scheduled
            self._api_phase_wrappers.append((runner, 'execute_model', original_execute))
            install(runner, '_execute_model_generic', 'target_forward')
            install(runner, '_run_sampling', 'target_sampling')
            install(runner, '_sample_decode', 'decode_sampling_and_verification')
            install(runner.rejection_sampler, 'forward', 'target_verification')
            install(runner.drafter.model, 'forward', 'draft_tensor_program')
            install(runner.drafter, 'observe_prefill', 'draft_prefill_context')
            return {'rank': self.rank, 'enabled': True}
        for owner, name, original in getattr(self, '_api_phase_wrappers', []):
            setattr(owner, name, original)
        self._api_phase_wrappers = []
        return {'rank': self.rank, 'scope': 'Host call spans; optional device synchronization; diagnostic only',
                'synchronized': synchronize, 'events': getattr(self, '_api_phase_events', []),
                'phases': {name: {'count': len(rows), 'total_ms': sum(rows),
                                 'p50_ms': statistics.median(rows), 'max_ms': max(rows)}
                           for name, rows in getattr(self, '_api_phase_times', {}).items() if rows}}

    def framework_dflash_audit(self, enabled, max_calls=3):
        """Compare captured proposals with the uncaptured checkpoint program.

        Recommitting the same accepted rows is idempotent. This hook is only
        for a loopback validation run and is removed before latency measurement.
        """
        if enabled not in ('on', 'off') or type(max_calls) is not int or not 1 <= max_calls <= 8:
            raise ValueError('Explicit audit mode and 1..8 calls required')
        import inspect
        if enabled == 'on':
            if getattr(self, '_dflash_audit_original', None):
                raise ValueError('Audit already active')
            model = self.model_runner.drafter.model
            original = model.forward
            raw = inspect.getclosurevars(original).nonlocals['orig_fwd']
            self._dflash_audit_original = (model, original)
            self._dflash_audit_results = []

            @wraps(original)
            def audited(*args, **kwargs):
                actual = original(*args, **kwargs)
                if len(self._dflash_audit_results) >= max_calls:
                    return actual
                import torch
                import habana_frameworks.torch.core as hc
                hc.mark_step()
                actual_cpu = actual.cpu().clone()
                context = self.model_runner.drafter.context
                positions_before = context.storage.positions.cpu().clone()
                expected = raw(*args, **kwargs)
                hc.mark_step()
                expected_cpu = expected.cpu()
                positions_after = context.storage.positions.cpu()
                record = dict(pass_proposals=torch.equal(actual_cpu, expected_cpu),
                              pass_positions=torch.equal(positions_before, positions_after),
                              actual=actual_cpu.tolist(), expected=expected_cpu.tolist(),
                              query_positions=args[1].cpu().tolist(), counts=args[2].cpu().tolist(),
                              next_positions=args[5].cpu().tolist())
                self._dflash_audit_results.append(record)
                if not record['pass_proposals'] or not record['pass_positions']:
                    raise RuntimeError('Captured DFlash diverges from uncaptured checkpoint program')
                return actual

            model.forward = audited
            return dict(rank=self.rank, enabled=True)
        current = getattr(self, '_dflash_audit_original', None)
        if current:
            current[0].forward = current[1]
            self._dflash_audit_original = None
        return dict(rank=self.rank, cases=getattr(self, '_dflash_audit_results', []))
