"""Pinned batch SWA registration sharing one TPC database with the M1 AV path."""
from pathlib import Path
import os
from gaudi_kernels.engine.context import context

_LIBRARIES = None


def manifest():
    ctx = context()
    libs = {'tpc':ctx.artifact('swa_bundle','tpc').verify(),
            'torch':ctx.artifact('swa_batch','torch').verify()}
    registered = {str(Path(p).resolve()) for p in os.environ.get('GC_KERNEL_PATH','').split(':') if p}
    if str(libs['tpc'].path) not in registered:
        raise ValueError('Shared SWA TPC database missing')
    return {k:{'path':str(v.path),'sha256':v.sha256} for k,v in libs.items()}


def prepare():
    global _LIBRARIES
    import torch
    libraries = manifest()
    if _LIBRARIES is not None and _LIBRARIES != libraries:
        raise ValueError('Cannot replace loaded batch SWA libraries')
    if _LIBRARIES is None:
        torch.ops.load_library(libraries['torch']['path'])
        getattr(torch.ops.gaudi_swa128_batch,'window_fp32')
        getattr(torch.ops.gaudi_swa128_batch,'window_quad_fp32')
        _LIBRARIES = libraries
    return snapshot()


def validate_policy():
    if _LIBRARIES is None or _LIBRARIES != manifest():
        raise ValueError('Batch SWA was not explicitly prepared')


def snapshot():
    return {'libraries':_LIBRARIES, 'scope':'Operator byte identity and prior component gates; full-model admission separate'}
