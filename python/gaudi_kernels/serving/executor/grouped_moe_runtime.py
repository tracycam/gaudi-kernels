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
_CAPTURES = {}


def validate_rows(rows):
    if (type(rows) is not tuple or tuple(sorted(set(rows))) != rows or
            any(type(t) is not int or t not in (512, 513, 1024, 2048, 4096) for t in rows)):
        raise ConfigError('Unsupported grouped MoE static shapes')


def plan_for(rows):
    validate_rows((rows,))
    if rows <= 513:
        return ExpertPlan(1, workspace_budget=2*1024**3, row_caps=(64, rows))
    caps = tuple(sorted({64, 256, 1024, rows}))
    return ExpertPlan(1, down_n_tile=512, workspace_budget=3*1024**3, row_caps=caps)


def prepare():
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
        for op in ('count', 'reshape', 'batch_mm', 'prefix', 'decode', 'combine'):
            getattr(torch.ops.gaudi_expert_partition, op)
        _LIBRARIES = libraries
    return {'libraries': _LIBRARIES}


def forward(x, ids, routing, gp, gs, down, ds, table, directions):
    import torch
    from gaudi_kernels.moe_expert_dispatch import expert_moe
    from .batch_ops import moe
    if _LIBRARIES is None:
        raise ConfigError('Grouped MoE must be prepared before model capture')
    if ids.shape[1] > 8:
        # General model adapter accepts larger top-k; this backend is bounded.
        return moe(x, ids, routing, gp, gs, down, ds, table, directions, mode='broadcast')
    # Lazy CustomOp inputs must have producers before the recipe is captured.
    # Keep materialization at the entry, never once per expert or bucket.
    key=str(x.shape[0])
    _CAPTURES[key]=_CAPTURES.get(key,0)+1
    x = x.contiguous().clone()
    ids = ids.to(torch.int32).contiguous().clone()
    routing = routing.to(torch.float32).contiguous().clone()
    return expert_moe(x, ids, routing, gp, gs, down, ds, table, directions,
                      plan=plan_for(x.shape[0]), tpc=moe, decoder='k8',
                      empty_mode=1, padding_mode=1)


def reset_counters():
    _CAPTURES.clear()


def snapshot():
    return {'libraries':_LIBRARIES,'graph_body_calls_by_rows':dict(_CAPTURES),
            'scope':'Python graph construction calls; not device execution counts'}
