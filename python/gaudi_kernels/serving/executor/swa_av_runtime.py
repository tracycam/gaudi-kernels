"""Default-off, separately pinned SWA AV-hoist library registration; no device IO."""
import hashlib
import os
from pathlib import Path
import re
_LIBRARIES = {}

def library_manifest():
    from gaudi_kernels.engine.context import context as execution_context
    ctx = execution_context()
    if ctx.has('swa_bundle'):
        libraries = {k:{'path':str(v.path),'sha256':v.sha256} for k,v in {
            'tpc':ctx.artifact('swa_bundle','tpc').verify(),
            'torch':ctx.artifact('swa_av','torch').verify()}.items()}
    else:
        libraries = ctx.libraries('swa_av')
    registered = {str(Path(p).resolve()) for p in os.getenv('GC_KERNEL_PATH', '').split(':') if p}
    if libraries['tpc']['path'] not in registered:
        raise ValueError('TPC library absent from GC_KERNEL_PATH')
    return libraries

def prepare():
    """Called after the old shape bridge is loaded, before model capture."""
    global _LIBRARIES
    libraries = library_manifest()
    if _LIBRARIES:
        if libraries != _LIBRARIES:
            raise ValueError('Cannot replace loaded SWA AV-hoist libraries inside a running process')
        return snapshot()
    import torch
    torch.ops.load_library(libraries['torch']['path'])
    getattr(torch.ops.gaudi_swa128_avgroup, 'group')
    _LIBRARIES = libraries
    return snapshot()

def validate_policy(policy):
    if policy not in ('vendor', 'fp32_fast', 'fp32_quad', 'fp32_av_hoist', 'fp32_batch', 'fp32_batch_quad'):
        raise ValueError('Unknown production SWA policy')
    if policy in ('fp32_av_hoist', 'fp32_batch', 'fp32_batch_quad'):
        if not _LIBRARIES or library_manifest() != _LIBRARIES:
            raise ValueError('SWA AV-hoist libraries were not explicitly prepared')

def snapshot():
    return {'libraries_prepared': bool(_LIBRARIES), 'libraries': {k: dict(v) for (k, v) in _LIBRARIES.items()}, 'policy_default': 'vendor', 'automatic_short_context_switch': False, 'scope': 'opt-in instruction schedule; Python capture counts live in the SWA installer; device/model gates remain separate'}
