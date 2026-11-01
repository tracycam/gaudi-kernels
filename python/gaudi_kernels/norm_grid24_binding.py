from gaudi_kernels.engine.context import context as execution_context
"""Exact qualified producer/reference pins; no environment-controlled approval."""
import os
from pathlib import Path

PINS = {
    ('norm_grid24', 'tpc'): ('e8487fcb1f0e58cbab74d5dd49dcc4441db313532dd251fca3499866c1812ebd',
                             '1aaa32187c9cd30a57f6d448867823be9356fc73385b094abc4281728d92e17b',
                             'd0e16def1133c7c36141958056db1fce1342b73b6d43b4d1706a80e575b6b6f1',
                             '517dc7aa8a6b91974e063e776a24c66c68e76dbc09b63c5621c3065669b519b0',
                             '24c3d70da960897d1e5cd7d5c8bfbcbf36f7c791557fe5bc4257545b9b29d828',
                             'f051793cbdecd40bcee76de80a9f404ba69b599f99bbb7cec2dcb41ec7c9446d',
                             '2b4e41e153d38d16484564682d3f28e472b1ad818a4ce815678dd308c579dace'),
    ('norm_grid24', 'torch'): ('ebee484a85fd44fb06c9b123e0bd3b63b01bab7c117fd06527af05faf0ef5976',
                               '248b19f9c06db23f62c70a5c4d8272e85a7f17caead9c0ffbc7a758db29bd9a0',
                               '8732afec4808539fbe13fac9fafa49001f1ff45b43edbfb4a629fcb17c80ab23',
                               '0f881ce65b6c8fc40529cc982aa4c84dee8f84d4c54e2c81f0edb6a8880bed8c'),
    ('norm', 'tpc'): ('39729fbc57737381d86b2fe640d1c4ce86b7f66f9b7e55567e5424fc4b17f882',
                      'd641a312ef694c178bc40d5e5804fb94ff61d6d4047ac4339e63be3356e606f4'),
    ('norm', 'torch'): '329098542f9d1093b4ac01606d85e8511583c62d7fc54aef72ed478555a66968',
}


def verify(*, require_loaded=True):
    import torch
    loaded={str(Path(p).resolve()) for p in torch.ops.loaded_libraries}
    databases={str(Path(p).resolve()) for p in os.getenv('GC_KERNEL_PATH','').split(':') if p}
    record={}
    for key,expected in PINS.items():
        artifact=execution_context().artifact(*key).verify()
        path=artifact.path
        actual=artifact.sha256
        if not artifact.qualifies(expected):raise ValueError('unqualified norm producer/reference: '+'.'.join(key))
        if require_loaded and str(path) not in (loaded if key[1] == 'torch' else databases):
            raise ValueError('qualified norm library is not loaded/configured: '+'.'.join(key))
        record['.'.join(key)]={'path':str(path),'sha256':actual}
    return {'contract':'grid24_residual_norm_block128_v1','guid':'gk_norm_block128_grid24_v1',
            'libraries':record,'scope':'M1 H6144 residual present; source/ELF/device gate binding, not model acceptance'}
