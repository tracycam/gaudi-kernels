"""Default-off post-vendor-TopK hook; no import-time loading or policy changes."""
from gaudi_kernels.engine.context import context as execution_context
import hashlib
import math
import os
from pathlib import Path
_POLICY = 'vendor'
_POST = None
_LIBRARIES = {}
_CALLS = {'vendor': 0, 'scalar': 0, 'vector': 0}
_APPLIED = {'scalar': 0, 'vector': 0}
_FALLBACK = {}

def prepare():
    """Load explicitly supplied, frozen libraries before any model capture."""
    global _POST, _LIBRARIES
    if not execution_context().has('router_post'):
        return snapshot()
    if _POST is not None:
        return snapshot()
    libraries = execution_context().libraries('router_post')
    registered = {str(Path(p).resolve()) for p in os.getenv('GC_KERNEL_PATH', '').split(':') if p}
    if libraries['tpc']['path'] not in registered:
        raise ValueError('Frozen router TPC library is missing from GC_KERNEL_PATH')
    import torch
    from gaudi_kernels.experimental_router_post import post_top8
    torch.ops.load_library(libraries['torch']['path'])
    (_POST, _LIBRARIES) = (post_top8, libraries)
    return snapshot()

def validate_policy(policy):
    if policy not in _CALLS:
        raise ValueError('Unknown post-TopK policy')
    if policy != 'vendor' and (not execution_context().has('router_post') or _POST is None):
        raise ValueError('Post-TopK libraries were not explicitly prepared')

def set_policy(policy):
    """Caller must drain native replay and clear HPU graphs before this call."""
    global _POLICY
    validate_policy(policy)
    _POLICY = policy
    for counts in (_CALLS, _APPLIED):
        for key in counts:
            counts[key] = 0
    _FALLBACK.clear()
    return snapshot()

def _fallback(reason):
    _FALLBACK[reason] = _FALLBACK.get(reason, 0) + 1
    return None

def try_post(scores, ordered_ids, renormalize, factor):
    """Consume the caller's existing vendor IDs. None means unchanged fallback."""
    _CALLS[_POLICY] += 1
    if _POLICY == 'vendor':
        return None
    import torch
    if tuple(scores.shape) != (1, 384) or tuple(ordered_ids.shape) != (1, 8):
        return _fallback('shape')
    if scores.device.type != 'hpu' or scores.device != ordered_ids.device or scores.dtype != torch.float32 or (ordered_ids.dtype not in (torch.int32, torch.int64)):
        return _fallback('device_or_dtype')
    if not scores.is_contiguous() or not ordered_ids.is_contiguous() or scores.requires_grad:
        return _fallback('layout_or_autograd')
    if not math.isfinite(factor) or abs(factor) > 3.4028234663852886e+38:
        return _fallback('factor')
    result = _POST(scores, ordered_ids, renormalize, factor, variant=_POLICY)
    _APPLIED[_POLICY] += 1
    return result

def snapshot():
    return {'policy': _POLICY, 'libraries_prepared': _POST is not None, 'libraries': {key: dict(value) for (key, value) in _LIBRARIES.items()}, 'python_capture_calls': dict(_CALLS), 'applied_capture_calls': dict(_APPLIED), 'fallbacks': dict(_FALLBACK), 'scope': 'Python dispatch/capture counts; cached device replays are not counted', 'device_qualification': 'separate module and model gates required'}
