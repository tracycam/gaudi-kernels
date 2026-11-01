"""Apply installed operator policies; graph execution belongs to the bridge."""
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
    from gaudi_kernels.serving.executor.gp_scale_tail_runtime import prepare, validate_policy as validate_gp
    from gaudi_kernels.serving.executor.qkv_neumaier_isa_runtime import validate_policy as validate_reduce
    from gaudi_kernels.serving.executor.swa_av_runtime import validate_policy as validate_swa
    from gaudi_kernels.serving.executor.norm_grid24_runtime import validate_policy as validate_grid
    from gaudi_kernels.serving.executor.router_post_runtime import validate_policy as validate_router
    from gaudi_kernels.serving.executor.moe_sum_bf16_runtime import validate_policy as validate_sum
    from gaudi_kernels.serving.executor.moe_dispatch_runtime import DispatchState, dispatch_consumers
    runner = worker.model_runner
    prepared_dispatch = DispatchState(d.moe.dispatch)
    dispatch_consumers(runner.model)
    if d.moe.gate_up.impl == 'tpc_folded_scale_tail':
        prepare()
        validate_gp()
    if d.qkv.reduce == 'neumaier_isa':
        validate_reduce()
    if d.attention.swa in ('fp32_av_hoist','fp32_batch','fp32_batch_quad'):
        validate_swa(d.attention.swa)
    if d.attention.swa in ('fp32_batch','fp32_batch_quad'):
        from gaudi_kernels.serving.executor.swa_batch_runtime import validate_policy as validate_batch
        validate_batch()
    validate_grid(d.norm.impl == 'fp32_grid24_quant')
    validate_router('vector' if d.moe.router.post == 'vector_top8' else 'vendor')
    sum_policy = 'moe_final_bf16' if d.moe.combine.sum_dtype == 'bf16' else 'baseline'
    validate_sum(sum_policy)
    if d.moe.dispatch.grouped_rows:
        from gaudi_kernels.serving.executor.grouped_moe_runtime import prepare as prepare_grouped, validate_rows as validate_grouped
        validate_grouped(d.moe.dispatch.grouped_rows)
        prepare_grouped(require_largest=bool(d.moe.dispatch.grouped_slot_budget or d.moe.dispatch.grouped_padded_row_budget))
    required = ['block_fp8', 'norm', 'swa', 'qkv_post', 'folded', 'down', 'router_post']
    if any((not runtime.has(name) for name in required)):
        raise ValueError('Incomplete qualified installation')
    hc.mark_step()
    torch.hpu.synchronize()
    clear = getattr(runner.model, 'clear_cache', None)
    if not callable(clear):
        raise RuntimeError('No model graph cache invalidator')
    draft = getattr(getattr(runner, 'drafter', None), 'model', None)
    draft_clear = getattr(draft, 'clear_cache', None)
    if getattr(runner, 'speculative_config', None) is not None and (not callable(draft_clear)):
        raise RuntimeError('No drafter graph cache invalidator')
    clear()
    from gaudi_kernels.serving.executor.grouped_moe_runtime import clear_cache as clear_grouped_cache
    clear_grouped_cache()
    # The caller has drained device work above. Retire whole-step captures with
    # the forward cache when arithmetic changes; object IDs are not cache keys.
    getattr(runner, '_step_graph_programs', {}).clear()
    runner._step_graph_sample = None
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
    from gaudi_kernels.serving.executor.moe_dispatch_runtime import install_state
    runner._moe_dispatch_state = install_state(runner.model, d.moe.dispatch, prepared=prepared_dispatch)
    dispatch = runner._moe_dispatch_state.snapshot()
    from gaudi_kernels.serving.executor.moe_sum_bf16_runtime import set_policy as set_sum
    moe_sum = set_sum(sum_policy)
    runtime.selection = selected
    runner._production_config = cfg
    runner._production_selected_policy = cfg.arithmetic_fingerprint
    runner._policy_generation = getattr(runner, '_policy_generation', 0) + 1
    return {'rank': worker.rank, 'policy': cfg.arithmetic_fingerprint, 'resolved_config': selected.to_dict(), 'details': result, 'norm': norm, 'swa': swa, 'gp': gp_snapshot(), 'down': down_snapshot(), 'collective_policy': runtime.collective_mode, 'qkv_postprocess': post, 'router_post': router, 'qkv_neumaier_isa': reducer_isa, 'moe_dispatch': dispatch, 'moe_sum_bf16': moe_sum, 'norm_grid24': grid}
