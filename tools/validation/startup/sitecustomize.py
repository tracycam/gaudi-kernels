"""Explicit framework validation boot; reduced layers are diagnostic-only."""
import os
if os.environ.get('GK_RUNTIME_MANIFEST'):
    try:
        from gaudi_kernels.engine.context import load_manifest
        runtime = load_manifest(os.environ['GK_RUNTIME_MANIFEST'])
        from tools.validation.partial_model import install as install_partial
        install_partial(runtime.startup.layers)
        from gaudi_kernels.serving.bootstrap_hooks import install as install_bindings
        install_bindings()
        from tools.validation.reference_hooks import install
        install()
        from tools.validation.core_timing import install as install_core_timing
        install_core_timing()
    except BaseException as error:
        raise SystemExit('Canonical bootstrap rejected: ' + str(error))
