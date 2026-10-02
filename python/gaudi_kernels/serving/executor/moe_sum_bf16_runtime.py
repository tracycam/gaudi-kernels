"""Default-off BF16 output for the existing final native-MoE FP32 AG sum.

No global communicator dtype change, tensor readback, synchronization, library
load or device operation. A one-use scope owns the exact Python input object
while the pinned MoERunner final reduction calls the existing communicator.
"""
from gaudi_kernels.engine.context import context as execution_context
from collections import Counter
from contextlib import contextmanager
import os
import threading
_POLICY = 'baseline'
_COUNTS = Counter()
_LOCAL = threading.local()

def validate_policy(policy):
    if policy not in ('baseline', 'moe_final_bf16'):
        raise ValueError('Unknown MoE final-sum policy')

def set_policy(policy):
    """Caller must drain replay and clear graph caches before changing policy."""
    global _POLICY
    validate_policy(policy)
    if getattr(_LOCAL, 'scope', None) is not None:
        raise RuntimeError('Cannot change MoE sum policy inside a reduction scope')
    _POLICY = policy
    _COUNTS.clear()
    return snapshot()

def eligible(runner, states, trunc_size, reduced, native):
    import torch
    config = runner.moe_config
    max_rows = execution_context().selection.engine.decode.tp_reduce.fp32_max_rows
    max_bytes = execution_context().selection.engine.decode.tp_reduce.fp32_max_bytes
    return native and (not reduced) and (not config.is_sequence_parallel) and (not config.skip_final_all_reduce) and (config.tp_size == 8) and (config.ep_size == 1) and (states.dtype == torch.float32) and (states.device.type == 'hpu') and states.ndim == 2 and states.shape[1] == 6144 and 1 <= states.shape[0] <= max_rows and states.numel()*states.element_size() <= max_bytes and states.is_contiguous() and (not states.requires_grad) and (trunc_size in (None, 6144))

@contextmanager
def final_scope(runner, states, trunc_size, reduced, native):
    if _POLICY != 'moe_final_bf16':
        yield False
        return
    if not eligible(runner, states, trunc_size, reduced, native):
        _COUNTS['ineligible_final_calls'] += 1
        yield False
        return
    if getattr(_LOCAL, 'scope', None) is not None:
        raise RuntimeError('Nested native MoE final-reduction scope is unsupported')
    scope = {'input': states, 'used': False}
    _LOCAL.scope = scope
    _COUNTS['eligible_final_calls'] += 1
    if trunc_size == 6144:
        _COUNTS['no_op_truncation_elided'] += 1
    try:
        yield True
    finally:
        if not scope['used']:
            _COUNTS['eligible_without_matching_ag'] += 1
        _LOCAL.scope = None

def select_bf16_output(input_tensor, ranks):
    """Called only inside the communicator's already selected AllGather path."""
    if _POLICY != 'moe_final_bf16':
        return False
    scope = getattr(_LOCAL, 'scope', None)
    if scope is None:
        return False
    if scope['used']:
        raise RuntimeError('A final MoE scope attempted more than one FP32 AG sum')
    if ranks != 8 or input_tensor is not scope['input']:
        _COUNTS['scope_identity_or_rank_mismatch'] += 1
        return False
    scope['used'] = True
    _COUNTS['sum_fb_capture_calls'] += 1
    return True

def snapshot():
    return {'policy': _POLICY, 'operator': 'precision_fix::sum_fb' if _POLICY == 'moe_final_bf16' else None, 'python_capture_counts': dict(_COUNTS), 'shape': ['1..'+str(execution_context().selection.engine.decode.tp_reduce.fp32_max_rows), 6144], 'tp': 8, 'scope': 'one-use native MoE final FP32 AG sum; same rank order then BF16 RNE; capture counts are not launches or device invocation counts'}
