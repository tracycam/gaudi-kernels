from gaudi_kernels.engine.context import context as execution_context
"""Exact qualified producer/reference pins; no environment-controlled approval."""
import hashlib
import os
from pathlib import Path

PINS = {
    ('norm_grid24', 'tpc'): 'e8487fcb1f0e58cbab74d5dd49dcc4441db313532dd251fca3499866c1812ebd',
    ('norm_grid24', 'torch'): 'ebee484a85fd44fb06c9b123e0bd3b63b01bab7c117fd06527af05faf0ef5976',
    ('norm', 'tpc'): '39729fbc57737381d86b2fe640d1c4ce86b7f66f9b7e55567e5424fc4b17f882',
    ('norm', 'torch'): '329098542f9d1093b4ac01606d85e8511583c62d7fc54aef72ed478555a66968',
}


def verify(*, require_loaded=True):
    import torch
    loaded={str(Path(p).resolve()) for p in torch.ops.loaded_libraries}
    databases={str(Path(p).resolve()) for p in os.getenv('GC_KERNEL_PATH','').split(':') if p}
    record={}
    for key,expected in PINS.items():
        path=Path(execution_context().path(*key)).resolve(strict=True)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('unqualified norm producer/reference: '+key)
        if require_loaded and str(path) not in (loaded if key[1] == 'torch' else databases):
            raise ValueError('qualified norm library is not loaded/configured: '+key)
        record['.'.join(key)]={'path':str(path),'sha256':actual}
    return {'contract':'grid24_residual_norm_block128_v1','guid':'gk_norm_block128_grid24_v1',
            'libraries':record,'scope':'M1 H6144 residual present; source/ELF/device gate binding, not model acceptance'}
