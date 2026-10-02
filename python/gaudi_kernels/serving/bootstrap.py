"""The single environment entry point needed by spawned vLLM workers."""
import ctypes
import os
from gaudi_kernels.engine.context import load_manifest

def start():
    manifest = os.environ.get('GK_RUNTIME_MANIFEST')
    if not manifest:
        return
    runtime = load_manifest(manifest)
    api = ctypes.CDLL(None)
    api.e1_configure_options.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
    api.e1_configure_options.restype = ctypes.c_int
    code = api.e1_configure_options(2 | 4, None)
    if code:
        raise RuntimeError('Explicit native startup options rejected: ' + str(code))
    from gaudi_kernels.serving import bootstrap_hooks
    return runtime
