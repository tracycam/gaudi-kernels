"""The single environment entry point needed by spawned vLLM workers."""
import os
from gaudi_kernels.engine.context import load_manifest

def start():
    manifest = os.environ.get('GK_RUNTIME_MANIFEST')
    if not manifest:
        return
    runtime = load_manifest(manifest)
    model = getattr(getattr(getattr(runtime, 'selection', None), 'engine', None), 'model', 'mimo-v2.6-pro')
    if model == 'mimo-v2.6-pro' and runtime.startup.layers != 70:
        raise ValueError('Production MiMo startup requires all 70 layers; use validation bootstrap for reduced models')
    from gaudi_kernels.serving import bootstrap_hooks
    bootstrap_hooks.install()
    return runtime
