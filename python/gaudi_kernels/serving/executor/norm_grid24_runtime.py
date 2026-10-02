"""Default-off independent norm producer registration, exact kernel-owned pins."""
from gaudi_kernels.engine.context import context as execution_context
import os
_BINDING = None

def prepare():
    global _BINDING
    if not execution_context().has('norm_grid24') or not execution_context().has('norm'):
        raise ValueError('grid24 requires explicit norm and grid24 enablement')
    from gaudi_kernels.norm_grid24_binding import verify
    candidate = verify(require_loaded=False)
    if _BINDING is not None:
        if candidate != _BINDING:
            raise ValueError('grid24 binding changed in running worker')
        return snapshot()
    import torch
    torch.ops.load_library(execution_context().path('norm_grid24', 'torch'))
    getattr(torch.ops.gaudi_norm_grid24, 'block128')
    _BINDING = verify()
    return snapshot()

def validate_policy(enabled):
    if enabled:
        from gaudi_kernels.norm_grid24_binding import verify
        if _BINDING is None or verify() != _BINDING:
            raise ValueError('grid24 producer has not been prepared with exact pins')

def snapshot():
    return {'prepared': _BINDING is not None, 'binding': _BINDING, 'default': 'separate'}
