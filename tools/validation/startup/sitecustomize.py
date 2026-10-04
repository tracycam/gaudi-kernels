"""Validation worker boot. The frozen shim is a diagnostic isolation arm only."""
import os
if os.environ.get('GK_RUNTIME_MANIFEST'):
    try:
        import ctypes
        api=ctypes.CDLL(None)
        import json
        from pathlib import Path
        selected = json.loads(Path(os.environ['GK_RUNTIME_MANIFEST']).read_text())
        framework = selected['selection']['engine'].get('runtime', {}).get('executor') == 'pytorch'
        if framework or hasattr(api,'e1_configure_options'):
            from gaudi_kernels.serving.bootstrap import start
            start()
        else:
            # This branch never exists in serving/startup. Admit only the exact
            # saved70-g shim to isolate source versus host-binary regressions.
            from gaudi_kernels.engine.context import load_manifest
            runtime=load_manifest(os.environ['GK_RUNTIME_MANIFEST'])
            pin='ba4bbe7f99dee06e91a1fdfe40d092bb1b6505fd13891731a776e229f2a1921f'
            if runtime.sha('replay','host')!=pin:raise ValueError('Unqualified historical shim')
            os.environ['E1_ALLOW_SYNC']='1';os.environ['E1_ALLOW_MEMCPY']='1'
            from gaudi_kernels.serving import bootstrap_hooks
        from tools.validation.reference_hooks import install
        install()
        from tools.validation.core_timing import install as install_core_timing
        install_core_timing()
    except BaseException as error:
        raise SystemExit('Canonical bootstrap rejected: ' + str(error))
