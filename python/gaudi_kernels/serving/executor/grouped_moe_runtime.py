"""Production adapter for original-storage expert GEMM, selected at capture.

Small token shapes retain the existing TPC path. This adapter owns no weights
and reads no device metadata on the host. The recipe partitions experts by M.
"""
from pathlib import Path
import os
from gaudi_kernels.engine.config import ConfigError
from gaudi_kernels.engine.context import context
from gaudi_kernels.moe_expert_plan import ExpertPlan

_LIBRARIES = None
_GRAPH_CACHE = {}


def clear_cache():
    _GRAPH_CACHE.clear()


def validate_rows(rows):
    if (type(rows) is not tuple or tuple(sorted(set(rows))) != rows or
            any(type(t) is not int or not 16 <= t <= 4096 for t in rows)):
        raise ConfigError('Unsupported grouped MoE static shapes')


def plan_for(rows, *, min_mme_rows=1, total_slot_budget=0, total_padded_row_budget=0):
    validate_rows((rows,))
    if rows <= 513:
        b0 = min(rows, max(16, (64 * rows + 511) // 512))
        caps = tuple(n for n in sorted({b0, rows}) if n >= min_mme_rows)
        return ExpertPlan(min_mme_rows, down_n_tile=2048, workspace_budget=2*1024**3, row_caps=caps,
                          total_slot_budget=total_slot_budget,
                          total_padded_row_budget=total_padded_row_budget)
    b0 = min(rows, max(4, (16 * rows + 511) // 512))
    b1 = min(rows, max(16, (64 * rows + 447) // 512))
    caps = tuple(n for n in sorted({b0, b1, rows}) if n >= min_mme_rows)
    if not total_slot_budget and not total_padded_row_budget and min_mme_rows == 1:
        total_padded_row_budget = 384 * b0 + 2 * max(384 * b0, 96 * b1, 12 * rows)
    return ExpertPlan(min_mme_rows, down_n_tile=2048, workspace_budget=5*1024**3, row_caps=caps,
                      total_slot_budget=total_slot_budget, total_padded_row_budget=total_padded_row_budget)


def prepare(*, require_largest=False):
    global _LIBRARIES
    import torch
    ctx = context()
    keys = ('moe_bundle.tpc', 'expert.torch', 'moe_batch_provider.host', 'moe_expert_provider.host')
    libraries = {}
    for key in keys:
        name, kind = key.split('.')
        artifact = ctx.artifact(name, kind).verify()
        libraries[key] = {'path':str(artifact.path), 'sha256':artifact.sha256}
    registered = {str(Path(p).resolve()) for p in os.environ.get('GC_KERNEL_PATH', '').split(':') if p}
    if libraries['moe_bundle.tpc']['path'] not in registered:
        raise ConfigError('Grouped MoE requires the shared registered TPC provider')
    if _LIBRARIES is not None and _LIBRARIES != libraries:
        raise ConfigError('Cannot replace loaded grouped MoE libraries')
    if _LIBRARIES is None:
        torch.ops.load_library(libraries['expert.torch']['path'])
        for op in ('count', 'reshape', 'batch_mm', 'prefix', 'prefix_largest', 'decode', 'combine', 'grouped_forward'):
            getattr(torch.ops.gaudi_expert_partition, op)
        _LIBRARIES = libraries
    if require_largest:
        getattr(torch.ops.gaudi_expert_partition, 'prefix_largest')
    return {'libraries': _LIBRARIES}


def _execute(x, ids, routing, gp, gs, down, ds, table, directions, plan, moe):
    import torch
    from gaudi_kernels.moe_expert_dispatch import expert_moe
    # Lazy CustomOp inputs must have producers before the recipe is captured.
    # Keep materialization at the entry, never once per expert or bucket.
    x = x.contiguous().clone()
    ids = ids.to(torch.int32).contiguous().clone()
    routing = routing.to(torch.float32).contiguous().clone()
    return expert_moe(x, ids, routing, gp, gs, down, ds, table, directions,
                      plan=plan, tpc=moe, decoder='k8',
                      empty_mode=1, padding_mode=1, combine_mode='merged')


def forward(x, ids, routing, gp, gs, down, ds, table, directions, *, plan=None, use_graph=False):
    from .batch_ops import moe
    if _LIBRARIES is None:
        raise ConfigError('Grouped MoE must be prepared before model capture')
    if ids.shape[1] > 8:
        # General model adapter accepts larger top-k; this backend is bounded.
        return moe(x, ids, routing, gp, gs, down, ds, table, directions, mode='broadcast')
    resolved_plan = plan_for(int(x.shape[0])) if plan is None else plan
    if use_graph and getattr(getattr(x, 'device', None), 'type', None) == 'hpu':
        import habana_frameworks.torch as htorch
        import habana_frameworks.torch.core as htcore
        if htorch.utils.internal.is_lazy():
            key = (int(x.shape[0]), resolved_plan, 0 if directions is None else int(directions.data_ptr()))
            graphed = _GRAPH_CACHE.get(key)
            if graphed is None:
                def _run(x_, ids_, routing_, gp_, gs_, down_, ds_, table_):
                    return _execute(x_, ids_, routing_, gp_, gs_, down_, ds_, table_, directions, resolved_plan, moe)
                graphed = htorch.hpu.wrap_in_hpu_graph_func(
                    _run, disable_tensor_cache=True, dry_run=False, free_inplace=False)
                _GRAPH_CACHE[key] = graphed
            htcore.mark_step()
            out = graphed(x, ids, routing, gp, gs, down, ds, table).clone()
            htcore.mark_step()
            return out
    return _execute(x, ids, routing, gp, gs, down, ds, table, directions, resolved_plan, moe)
