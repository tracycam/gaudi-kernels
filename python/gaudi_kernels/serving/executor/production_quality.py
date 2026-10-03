"""Explicit model-quality diagnostics, disabled during performance measurements.

Capture complete logits after a normal bridge invocation. A fixed prompt plus
the reference continuation gives both policies identical teacher-forced input.
This diagnostic adds synchronization/readback and must never count as TPS.
"""
from gaudi_kernels.engine.context import context as execution_context
import os
from pathlib import Path

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
                uploaded = self._native_transfers.audit_inputs
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
                capturing = bool(getattr(self, '_native_backend', {}).get('capturing', False))
                with observing(execution_context().startup.inventory and (warmup or capturing), 'capture' if capturing else 'warmup'):
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
            uploaded = self._native_transfers.audit_inputs
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
    if execution_context().native.active:
        raise ValueError('Quality capture requires the bridge; disable native replay first')
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
    transfers = getattr(worker.model_runner, '_native_transfers', None)
    if audit_mode in ('batch','batch_boundaries') and transfers is None:
        raise ValueError('Named batch diagnostics require the owned runner')
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
from gaudi_kernels.engine.inventory import track_selection

@track_selection
def configure_policy(worker, document):
    """Apply typed arithmetic choices only after draining and cache invalidation."""
    from gaudi_kernels.engine.selection import PolicySelection
    from gaudi_kernels.engine.context import context as execution_context
    import torch
    import habana_frameworks.torch.core as hc
    from gaudi_kernels.production_integration import set_policy, set_reduction_policy
    selected = PolicySelection.from_dict(document)
    cfg = selected.engine
    d = cfg.decode
    runtime = execution_context()
    if runtime.native.active:
        raise ValueError('Drain/disable native replay before changing arithmetic policy')
    from gaudi_kernels.serving.executor.gp_scale_tail_runtime import prepare, validate_policy as validate_gp
    from gaudi_kernels.serving.executor.qkv_neumaier_isa_runtime import validate_policy as validate_reduce
    from gaudi_kernels.serving.executor.swa_av_runtime import validate_policy as validate_swa
    from gaudi_kernels.serving.executor.norm_grid24_runtime import validate_policy as validate_grid
    from gaudi_kernels.serving.executor.router_post_runtime import validate_policy as validate_router
    from gaudi_kernels.serving.executor.moe_sum_bf16_runtime import validate_policy as validate_sum
    from gaudi_kernels.serving.executor.moe_dispatch_runtime import validate_rows
    if d.moe.gate_up.impl == 'tpc_folded_scale_tail':
        prepare()
        validate_gp()
    if d.qkv.reduce == 'neumaier_isa':
        validate_reduce()
    if d.attention.swa in ('fp32_av_hoist','fp32_batch','fp32_batch_quad'):
        validate_swa(d.attention.swa)
    if d.attention.swa in ('fp32_batch','fp32_batch_quad'):
        from .swa_batch_runtime import validate_policy as validate_batch
        validate_batch()
    validate_grid(d.norm.impl == 'fp32_grid24_quant')
    validate_router('vector' if d.moe.router.post == 'vector_top8' else 'vendor')
    sum_policy = 'moe_final_bf16' if d.moe.combine.sum_dtype == 'bf16' else 'baseline'
    validate_sum(sum_policy)
    validate_rows(d.moe.dispatch.compact_rows)
    if d.moe.dispatch.grouped_rows:
        from .grouped_moe_runtime import prepare as prepare_grouped, validate_rows as validate_grouped
        validate_grouped(d.moe.dispatch.grouped_rows)
        prepare_grouped()
    required = ['block_fp8', 'norm', 'swa', 'qkv_post', 'folded', 'down', 'router_post']
    if any((not runtime.has(name) for name in required)):
        raise ValueError('Incomplete qualified installation')
    hc.mark_step()
    torch.hpu.synchronize()
    runner = worker.model_runner
    if hasattr(runner, '_native_reset'):
        runner._native_reset()
    clear = getattr(runner.model, 'clear_cache', None)
    if not callable(clear):
        raise RuntimeError('No model graph cache invalidator')
    draft = getattr(getattr(runner, 'drafter', None), 'model', None)
    draft_clear = getattr(draft, 'clear_cache', None)
    if getattr(runner, 'speculative_config', None) is not None and (not callable(draft_clear)):
        raise RuntimeError('No drafter graph cache invalidator')
    clear()
    if callable(draft_clear):
        draft_clear()
    set_policy(selected.block_policy)
    result = set_reduction_policy('neumaier_isa_fp32' if d.qkv.reduce == 'neumaier_isa' else 'sequential')
    from gaudi_kernels.serving.executor.qkv_neumaier_isa_runtime import record_selection
    reducer_isa = record_selection(d.qkv.reduce == 'neumaier_isa')
    runtime.collective_mode = 'auto_fp32' if d.tp_reduce.collective == 'all_gather_fp32_local_sum' else 'auto'
    from gaudi_kernels.serving.executor.precision_runtime import COUNTS
    COUNTS['f32_gather_f32_sum'] = 0
    runner._production_cpu_reference = selected.reference != 'device'
    from gaudi_kernels.vllm_norm import prepare_vllm_norm, set_norm_policy
    norm = {'prepared': prepare_vllm_norm(runner.model), 'policy': set_norm_policy('vendor' if d.norm.impl == 'vendor' else 'fp32')}
    if not norm['prepared']['matched']:
        raise RuntimeError('No matching model norms')
    from gaudi_kernels.vllm_swa_install import set_swa_policy, prepare_vllm_swa, snapshot_swa_counters
    snapshot_swa_counters(reset=True)
    set_swa_policy(d.attention.swa)
    swa = prepare_vllm_swa(runner.model)
    from gaudi_kernels.serving.executor.swa_av_runtime import snapshot as av_snapshot
    swa['av_libraries'] = av_snapshot()
    if d.attention.swa != 'vendor' and (not swa['matched']):
        raise RuntimeError('No matching SWA layers')
    from gaudi_kernels.vllm_qkv_postprocess import install_vllm_qkv_postprocess, snapshot_qkv_postprocess
    snapshot_qkv_postprocess(reset=True)
    post_policy = 'vendor' if d.qkv.post == 'vendor' else 'fused'
    scope = 'swa128_or_full' if d.qkv.post == 'fused_all' else 'swa128'
    post = install_vllm_qkv_postprocess(post_policy, model=runner.model, attention_scope=scope,max_rows=d.qkv.post_max_rows)
    if post_policy == 'fused' and (not post['prepared']['matched']):
        raise RuntimeError('No matching postprocess layers')
    if scope == 'swa128_or_full' and (not post['prepared'].get('matched_by_attention_kind', {}).get('full')):
        raise RuntimeError('No matching full-attention postprocess layers')
    from gaudi_kernels.vllm_norm_grid24 import install as install_grid, snapshot as grid_snapshot
    grid_snapshot(reset=True)
    grid = install_grid('grid24' if d.norm.impl == 'fp32_grid24_quant' else 'separate', model=runner.model)
    if d.norm.impl == 'fp32_grid24_quant' and (not grid['prepared']['matched']):
        raise RuntimeError('No eligible norm/QKV pairs')
    from gaudi_kernels.serving.executor.batch_ops import set_gp_policy, gp_snapshot, set_down_policy, down_snapshot
    gp_policy = {'legacy': 'legacy', 'tpc_folded': 'vector_fetch_folded', 'tpc_folded_scale_tail': 'vector_fetch_scale_tail'}[d.moe.gate_up.impl]
    set_gp_policy(gp_policy)
    set_down_policy('vector_fetch' if d.moe.down.impl == 'tpc_vector' else 'legacy')
    from gaudi_kernels.serving.executor.router_post_runtime import set_policy as set_router
    router = set_router('vector' if d.moe.router.post == 'vector_top8' else 'vendor')
    from gaudi_kernels.serving.executor.moe_dispatch_runtime import set_rows
    dispatch = set_rows(d.moe.dispatch.compact_rows, d.moe.dispatch.grouped_rows)
    from gaudi_kernels.serving.executor.moe_sum_bf16_runtime import set_policy as set_sum
    moe_sum = set_sum(sum_policy)
    runtime.selection = selected
    runner._production_config = cfg
    runner._production_selected_policy = cfg.arithmetic_fingerprint
    return {'rank': worker.rank, 'policy': cfg.arithmetic_fingerprint, 'resolved_config': selected.to_dict(), 'details': result, 'norm': norm, 'swa': swa, 'gp': gp_snapshot(), 'down': down_snapshot(), 'collective_policy': runtime.collective_mode, 'qkv_postprocess': post, 'router_post': router, 'qkv_neumaier_isa': reducer_isa, 'moe_dispatch': dispatch, 'moe_sum_bf16': moe_sum, 'norm_grid24': grid}

def compare_logits(reference, candidate, *, contract='legacy'):
    import torch
    if contract not in ('legacy', 'fp32_arithmetic_v1'):
        raise ValueError('Unknown quality arithmetic contract')
    if reference.shape != candidate.shape:
        raise ValueError('Teacher logits shape mismatch')
    (reference, candidate) = (reference.double(), candidate.double())
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    if not finite:
        return {'pass': False, 'finite': False}
    (lp, lq) = (reference.log_softmax(-1), candidate.log_softmax(-1))
    kl = float((lp.exp() * (lp - lq)).sum(-1).max())
    relative_l2 = float((reference - candidate).norm() / reference.norm().clamp_min(1e-30))
    return {'pass': kl <= 0.01 and (contract == 'fp32_arithmetic_v1' or relative_l2 <= 0.02), 'finite': True, 'max_kl_ref_to_candidate': kl, 'relative_l2': relative_l2, 'top1_match_fraction': float((reference.argmax(-1) == candidate.argmax(-1)).double().mean()), 'gate': {'max_kl': 0.01, 'relative_l2': 0.02 if contract == 'legacy' else 'diagnostic'}, 'scope': 'full-vocabulary teacher-forced logits; bounded regression gate, not broad quality evaluation', 'contract': contract}

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
    result['moe_dispatch'] = moe_dispatch_snapshot()
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
    if None == '1' or execution_context().has('folded'):
        from gaudi_kernels.serving.executor.batch_ops import gp_snapshot
        result['gp'] = gp_snapshot()
    if execution_context().has('down'):
        from gaudi_kernels.serving.executor.batch_ops import down_snapshot
        result['down'] = down_snapshot()
    if execution_context().has('router_post'):
        from gaudi_kernels.serving.executor.router_post_runtime import snapshot as router_post_snapshot
        result['router_post'] = router_post_snapshot()
    return result
