"""Phase-independent token-row MoE. All routing stays on HPU; no data .item()."""
from gaudi_kernels.engine.context import context as execution_context
from pathlib import Path
import os
import torch
ROOT = Path(__file__).resolve().parent.parent
torch.ops.load_library(execution_context().path('batch'))
if execution_context().has('folded'):
    torch.ops.load_library(execution_context().path('folded', 'torch'))
if execution_context().has('scale_tail'):
    from gaudi_kernels.serving.executor.gp_scale_tail_runtime import prepare as prepare_gp_scale_tail
    prepare_gp_scale_tail()
_GP_POLICY = 'legacy'
_GP_COUNTS = {'legacy': 0, 'vector_fetch': 0, 'vector_fetch_folded': 0, 'vector_fetch_scale_tail': 0}
if execution_context().has('down'):
    torch.ops.load_library(execution_context().path('down', 'torch'))
_DOWN_POLICY = 'legacy'
_DOWN_COUNTS = {'legacy': 0, 'vector_fetch': 0}

def set_gp_policy(policy):
    global _GP_POLICY
    if policy not in _GP_COUNTS:
        raise ValueError('Unknown compact GP policy')
    if policy == 'vector_fetch' and None != '1':
        raise ValueError('Vector-fetch GP library was not installed')
    if policy == 'vector_fetch_folded' and (not execution_context().has('folded')):
        raise ValueError('Folded vector-fetch GP library was not installed')
    if policy == 'vector_fetch_scale_tail' or execution_context().has('scale_tail'):
        from gaudi_kernels.serving.executor.gp_scale_tail_runtime import record_selection
        record_selection(policy == 'vector_fetch_scale_tail')
    _GP_POLICY = policy

def gp_snapshot():
    state = {'policy': _GP_POLICY, 'python_capture_calls': dict(_GP_COUNTS), 'scope': 'Python compact GP dispatches; cached device replays are not counted'}
    if execution_context().has('scale_tail'):
        from gaudi_kernels.serving.executor.gp_scale_tail_runtime import snapshot
        state['scale_tail'] = snapshot()
    return state

def set_down_policy(policy):
    global _DOWN_POLICY
    if policy not in _DOWN_COUNTS:
        raise ValueError('Unknown compact down policy')
    if policy == 'vector_fetch' and (not execution_context().has('down')):
        raise ValueError('Vector-fetch down library was not installed')
    _DOWN_POLICY = policy

def down_snapshot():
    return {'policy': _DOWN_POLICY, 'python_capture_calls': dict(_DOWN_COUNTS), 'scope': 'Python compact down dispatches; cached device replays are not counted'}

@torch.library.register_fake('unified_batch::gp')
def _gp(w, s, x, table, ids):
    return x.new_empty((1, ids.numel() * 3 * 512), dtype=torch.float32)

@torch.library.register_fake('unified_batch::gate')
def _gate(partial, ids):
    return partial.new_empty((ids.numel(), 256), dtype=torch.bfloat16)

@torch.library.register_fake('unified_batch::down')
def _down(w, s, x, table, ids):
    return x.new_empty((1, ids.numel() * 6144), dtype=torch.float32)

@torch.library.register_fake('unified_batch::gate_broadcast')
def _broadcast(partial, ids):
    n = ids.numel() * 12
    return (partial.new_empty((n, 256), dtype=torch.bfloat16), ids.new_empty((1, n)))

@torch.library.register_fake('unified_batch::sorted_prep')
def _sorted(x, ids, permutation):
    n = ids.numel() * 3
    return (x.new_empty((n, 2048)), ids.new_empty((1, n)), ids.new_empty(ids.shape))

@torch.library.register_fake('unified_batch::combine')
def _combine(partial, routing, directions, inverse):
    return partial.new_empty((routing.shape[0], 6144))

@torch.library.register_fake('unified_batch::masked_gemv')
def _masked(w, s, x, table, mapping):
    return x.new_empty((1, x.shape[0] * 512), dtype=torch.float32)

def moe(x, ids, routing, gp, gs, dp, ds, table, directions, *, mode='broadcast', route_mask=None, debug=False):
    if x.ndim != 2 or x.shape[1] != 6144 or ids.ndim != 2 or (ids.shape != routing.shape) or (ids.shape[0] != x.shape[0]):
        raise ValueError('Expected x[T,6144] and ids/routing[T,topk]')
    if x.dtype != torch.bfloat16 or ids.shape[1] < 1 or ids.shape[1] > 384:
        raise ValueError('Expected BF16 activation and 1 <= top-k <= 384')
    if x.shape[0] == 0:
        if debug:
            raise ValueError('Empty batches have no intermediate debug tensors')
        return x.new_empty((0, 6144), dtype=torch.float32)
    if route_mask is not None:
        if route_mask.shape != ids.shape or route_mask.dtype != torch.bool:
            raise ValueError('route_mask must be bool[T,topk]')
        if mode == 'compact':
            mode = 'broadcast'
        ids = torch.where(route_mask, ids, torch.full_like(ids, -1))
        routing = torch.where(route_mask, routing, torch.zeros_like(routing))
    precise = '1' == '1'
    if precise and mode == 'sorted':
        raise ValueError('FP32 routing is certified for compact/broadcast only')
    x = x.clone()
    ids = ids.to(torch.int32).clone()
    routing = routing.to(torch.float32 if precise else torch.bfloat16).clone()
    if mode == 'compact':
        if _GP_POLICY == 'vector_fetch_scale_tail':
            from gaudi_kernels.serving.executor.moe_dispatch_runtime import scale_tail_row_allowed
            if not scale_tail_row_allowed(x.shape[0]) or ids.shape[1] != 8:
                raise ValueError('GP scale-tail is qualified for compact rows<=8 and top8 only')
            from gaudi_kernels.serving.executor.gp_scale_tail_runtime import operator
            gp_call = operator()
        else:
            gp_call = torch.ops.gaudi_activation_folded.gp if _GP_POLICY == 'vector_fetch_folded' else torch.ops.gaudi_gp_diagnostic.broadcast if _GP_POLICY == 'vector_fetch' else torch.ops.unified_batch.gp
        _GP_COUNTS[_GP_POLICY] += 1
        gp_out = gp_call(gp, gs, x, table, ids)
        gate = torch.ops.unified_batch.gate(gp_out, ids)
        _DOWN_COUNTS[_DOWN_POLICY] += 1
        down_call = torch.ops.gaudi_down_activation.broadcast if _DOWN_POLICY == 'vector_fetch' else torch.ops.unified_batch.down
        partial = down_call(dp, ds, gate, table, ids)
    else:
        if mode not in ('broadcast', 'sorted'):
            raise ValueError('Expected compact, broadcast or sorted')
        if mode == 'sorted':
            (values, permutation) = torch.sort(ids.reshape(-1))
            sorted_ids = values.reshape(ids.shape).clone()
            permutation = permutation.to(torch.int32).reshape(ids.shape).clone()
            (ax, mapping, inverse) = torch.ops.unified_batch.sorted_prep(x, sorted_ids, permutation)
            selected_ids = sorted_ids
        else:
            (ax, mapping) = torch.ops.native_mxfp4.prep(x, ids, 1, 3, ids.shape[1])
            selected_ids = ids
        gemv = torch.ops.unified_batch.masked_gemv if route_mask is not None else torch.ops.native_mxfp4.gemv
        gp_out = gemv(gp, gs, ax, table, mapping)
        (gate, down_map) = torch.ops.unified_batch.gate_broadcast(gp_out, selected_ids)
        partial = gemv(dp, ds, gate, table, down_map)
    if precise:
        from gaudi_kernels.serving.executor.precision_ops import combine
        out = combine(partial, routing, directions, 6144, ids.shape[1])
    else:
        out = torch.ops.unified_batch.combine(partial, routing, directions, inverse) if mode == 'sorted' else torch.ops.native_mxfp4.combine(partial, routing, directions, 6144, ids.shape[1])
    return (out, gp_out, gate, partial) if debug else out
