"""Explicit model-quality diagnostics, disabled during performance measurements.

Capture complete logits after a normal bridge invocation. A fixed prompt plus
the reference continuation gives both policies identical teacher-forced input.
This diagnostic adds synchronization/readback and must never count as TPS.
"""
from gaudi_kernels.engine.context import context as execution_context
import os
from pathlib import Path
from gaudi_kernels.serving.diagnostics.quality import compare_logits

def install_runner(module, *, runner_cls=None):
    cls = runner_cls or module.HPUModelRunner
    original = cls._execute_model_generic
    original_use_graphs = cls._use_graphs

    def use_graphs(self, *args, **kwargs):
        if getattr(self, '_production_boundary_session', None) is not None or getattr(self, '_production_cpu_reference', False) or getattr(self, '_production_same_input_audit', False) or (getattr(self, '_production_layer_diagnostics', None) is not None):
            return False
        return original_use_graphs(self, *args, **kwargs)

    def execute(self, *args, **kwargs):
        token_ids = args[0] if args else kwargs['token_ids']
        positions = args[1] if len(args) > 1 else kwargs['position_ids']
        metadata = args[2] if len(args) > 2 else kwargs['attn_metadata']
        stages = getattr(self, '_production_layer_diagnostics', None)
        staged_fp32 = getattr(self, '_production_same_input_audit', False) and 'fp32_arithmetic_v1' == 'fp32_arithmetic_v1'
        if staged_fp32:
            from gaudi_kernels.production_integration import set_same_input_audit_frame
            frame = None
            if token_ids.numel() == 1 and (not metadata.is_prompt):
                frame = {'input_ids': token_ids.detach().cpu().reshape(-1).tolist(), 'positions': positions.detach().cpu().reshape(-1).tolist()}
            set_same_input_audit_frame(frame)
        if stages is not None:
            stages.enabled = token_ids.numel() == 1 and (not metadata.is_prompt)
        try:
            session = getattr(self, '_production_boundary_session', None)
            batch_capture = getattr(self, '_production_capture_batch', False)
            expected = getattr(self, '_production_expected_requests', None) or self.input_batch.num_reqs
            if batch_capture and not metadata.is_prompt:
                uploaded = self._input_audit.audit_inputs
                groups = uploaded['block_groups'].reshape(-1).tolist()
                active_rows = sorted({int(v) for v in groups if v >= 0})
                layout=self._native_query_layout
                layout.validate(uploaded['positions'].reshape(-1).tolist(),active_rows,
                                uploaded['logits_indices'].reshape(-1).tolist())
                capture_rows = len(layout.request_ids) >= expected
            else:
                capture_rows = token_ids.numel() == 1
            capture = session is not None and capture_rows and not metadata.is_prompt
            if capture:
                with session.recording():
                    outputs = original(self, *args, **kwargs)
                session.flush(token_ids, positions)
                self._production_boundary_session = None
            else:
                from gaudi_kernels.engine.inventory import observing
                warmup = bool(kwargs.get('warmup_mode', False) or getattr(self, 'warmup_mode', False))
                with observing(execution_context().startup.inventory and warmup, 'warmup'):
                    outputs = original(self, *args, **kwargs)
            if stages is not None and stages.enabled:
                stages.flush(self._production_stage_path, token_ids, positions)
        finally:
            if stages is not None:
                stages.enabled = False
                stages.stack.clear()
            if staged_fp32:
                set_same_input_audit_frame(None)
        destination = getattr(self, '_production_quality_destination', None)
        if destination is not None and self.is_driver_worker:
            import torch
            logits = outputs[-1].detach().cpu().float()
            torch.save({'logits': logits, 'input_ids': token_ids.detach().cpu(), 'positions': positions.detach().cpu()}, destination)
        trace = getattr(self, '_production_quality_trace', None)
        if trace is not None and self.is_driver_worker and not metadata.is_prompt:
            import torch
            import hashlib
            import json
            uploaded = self._input_audit.audit_inputs
            active_rows = sorted({int(v) for v in uploaded['block_groups'].reshape(-1).tolist() if v >= 0})
            selected = uploaded['logits_indices'].reshape(-1).tolist()
            request_keys = []
            request_ids = []
            owners=self._native_query_layout.validate(uploaded['positions'].reshape(-1).tolist(),
                                                       active_rows,selected)
            for rid in owners:
                if rid is None:
                    request_keys.append(None)
                    request_ids.append(None)
                    continue
                req = self.requests[rid]
                request_ids.append(rid)
                request_keys.append(hashlib.sha256(json.dumps(req.prompt_token_ids).encode()).hexdigest())
            record = {'input_ids':uploaded['token_ids'], 'positions':uploaded['positions'],
                      'logits':outputs[-1].detach().cpu().float(),
                      'logits_indices':uploaded['logits_indices']}
            path = trace / (str(len(self._production_quality_frames))+'.pt')
            torch.save(record,path,pickle_protocol=2)
            self._production_quality_frames.append({'path':str(path),
                'input_ids':record['input_ids'].reshape(-1).tolist(),
                'positions':record['positions'].reshape(-1).tolist(),
                'logits_indices':record['logits_indices'].reshape(-1).tolist(),
                'logits_shape':list(record['logits'].shape),
                'active_query_rows':active_rows, 'request_keys':request_keys,
                'request_ids':request_ids,
                'request_count':self.input_batch.num_reqs})
        return outputs
    cls._execute_model_generic = execute
    cls._use_graphs = use_graphs

def configure_capture(worker, tag=None, audit_mode=None, expected_requests=None):
    import torch
    import habana_frameworks.torch.core as hc
    hc.mark_step()
    torch.hpu.synchronize()
    if audit_mode not in (None, 'plain', 'staged', 'boundaries', 'batch', 'batch_boundaries'):
        raise ValueError('Unknown quality capture mode')
    if expected_requests is not None and (audit_mode not in ('batch','batch_boundaries') or
            type(expected_requests) is not int or not 1 <= expected_requests <= 8):
        raise ValueError('Expected request count requires bounded batch diagnostics')
    path = None
    if tag is not None:
        if not tag or any((c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in tag)):
            raise ValueError('Invalid quality capture tag')
        root = Path(str(execution_context().startup.run_dir.parent)) / execution_context().startup.run_dir.name / 'quality-logits'
        root.mkdir(exist_ok=True)
        path = root / (tag + '.pt')
        if path.exists():
            raise FileExistsError(path)
    worker.model_runner._production_quality_destination = path
    worker.model_runner._production_boundary_session = None
    worker.model_runner._production_capture_batch = audit_mode in ('batch','batch_boundaries')
    transfers = getattr(worker.model_runner, '_input_audit', None)
    if audit_mode in ('batch','batch_boundaries') and transfers is None:
        from gaudi_kernels.serving.diagnostics.input_audit import InputAudit
        transfers = worker.model_runner._input_audit = InputAudit()
    if transfers is not None:
        transfers.audit_inputs = {} if audit_mode in ('batch','batch_boundaries') else None
    worker.model_runner._production_expected_requests = expected_requests
    captured_frames = list(getattr(worker.model_runner, '_production_quality_frames', []))
    worker.model_runner._production_quality_trace = None
    if audit_mode in ('batch','batch_boundaries'):
        if path is None:
            raise ValueError('Batch trace requires a unique tag')
        trace = path.parent / (tag+f'-frames-rank{worker.rank}')
        trace.mkdir(exist_ok=False)
        worker.model_runner._production_quality_trace = trace
        worker.model_runner._production_quality_frames = []
        worker.model_runner._production_quality_destination = None
        captured_frames = []
    if audit_mode in ('boundaries','batch_boundaries'):
        if path is None:
            raise ValueError('A boundary diagnostic needs a unique tag')
        from gaudi_kernels.serving.diagnostics.boundary_hashes import Session
        destination = path.with_name(path.stem + f'-boundaries-rank{worker.rank}.json')
        worker.model_runner._production_boundary_session = Session(worker.model_runner.model, execution_context().startup.layers, worker.rank, destination)
    stage_path = None
    stages = getattr(worker.model_runner, '_production_layer_diagnostics', None)
    if stages is not None:
        stages.close()
        worker.model_runner._production_layer_diagnostics = None
    if audit_mode is None and tag is not None and tag.endswith('-teacher-0') and execution_context().startup.layer_diagnostics:
        from gaudi_kernels.serving.diagnostics.layer_diagnostics import LayerDiagnostics
        stage_path = path.with_name(path.stem + f'-stages-rank{worker.rank}.pt')
        if stage_path.exists():
            raise FileExistsError(stage_path)
        worker.model_runner._production_layer_diagnostics = LayerDiagnostics(worker.model_runner.model, capture_rotary=not execution_context().has('qkv_post'))
        worker.model_runner._production_stage_path = stage_path
    audit = []
    if execution_context().startup.same_input_audit:
        from gaudi_kernels.production_integration import configure_same_input_audit
        audit = configure_same_input_audit(tag is not None and audit_mode == 'staged', contract='fp32_arithmetic_v1', tag=tag)
        worker.model_runner._production_same_input_audit = tag is not None and audit_mode == 'staged'
    from gaudi_kernels.production_integration import snapshot as block_snapshot
    grid_audit_layers = []
    if execution_context().has('norm_grid24'):
        from gaudi_kernels.vllm_norm_grid24 import snapshot as grid_snapshot
        grid_state = grid_snapshot()
        if grid_state['policy'] == 'grid24':
            grid_audit_layers = grid_state['expected_audit_layers']
    return {'rank': worker.rank, 'path': str(path) if path else None, 'stage_path': str(stage_path) if stage_path else None, 'same_input_qkv_audit': audit, 'capture_mode': audit_mode, 'selected_qkv_layers': list(block_snapshot().get('selected_layers', {})), 'expected_grid24_qkv_layers': grid_audit_layers, 'captured_forwards':captured_frames}
from gaudi_kernels.serving.policy import configure_policy

def snapshot(worker):
    from gaudi_kernels.engine.inventory import write_snapshot
    inventory_record = write_snapshot(worker)
    from gaudi_kernels.production_integration import snapshot as block_snapshot
    result = {'rank': worker.rank, 'block_fp8': block_snapshot(), 'runtime_inventory': inventory_record}
    manager = getattr(worker.model_runner, 'bucketing_manager', None)
    buckets = getattr(manager, 'decode_buckets', None)
    if buckets:
        # Request cardinality and compiled token rows differ (e.g. B3 -> T4).
        # Export the actual SDK declaration, not an assumed power-of-two rule.
        result['serving_decode_batch_buckets'] = sorted({int(bucket[0]) for bucket in buckets if bucket[0] > 0})
    if execution_context().has('norm_grid24'):
        from gaudi_kernels.vllm_norm_grid24 import snapshot as grid24_snapshot
        result['norm_grid24'] = grid24_snapshot()
    from gaudi_kernels.serving.executor.moe_dispatch_runtime import snapshot as moe_dispatch_snapshot
    result['moe_dispatch'] = moe_dispatch_snapshot(worker.model_runner._moe_dispatch_state)
    from gaudi_kernels.serving.executor.moe_sum_bf16_runtime import snapshot as moe_sum_snapshot
    result['moe_sum_bf16'] = moe_sum_snapshot()
    if execution_context().has('reduce_isa'):
        from gaudi_kernels.serving.executor.qkv_neumaier_isa_runtime import snapshot as isa_snapshot
        result['qkv_neumaier_isa'] = isa_snapshot()
    from gaudi_kernels.serving.executor.precision_runtime import COUNTS
    result['collective'] = {'policy': execution_context().collective_mode, 'fp32_ag_capture_calls': COUNTS.get('f32_gather_f32_sum', 0), 'scope': 'Python captures; native command list must independently prove HCCL calls'}
    if execution_context().has('swa'):
        from gaudi_kernels.vllm_swa_install import snapshot_swa_counters
        result['swa'] = snapshot_swa_counters()
        if execution_context().has('swa_av'):
            from gaudi_kernels.serving.executor.swa_av_runtime import snapshot as av_snapshot
            result['swa']['av_libraries'] = av_snapshot()
    if execution_context().has('qkv_post'):
        from gaudi_kernels.vllm_qkv_postprocess import snapshot_qkv_postprocess
        result['qkv_postprocess'] = snapshot_qkv_postprocess()
    if execution_context().has('folded'):
        from gaudi_kernels.serving.executor.batch_ops import gp_snapshot
        result['gp'] = gp_snapshot()
    if execution_context().has('down'):
        from gaudi_kernels.serving.executor.batch_ops import down_snapshot
        result['down'] = down_snapshot()
    if execution_context().has('router_post'):
        from gaudi_kernels.serving.executor.router_post_runtime import snapshot as router_post_snapshot
        result['router_post'] = router_post_snapshot()
    return result
