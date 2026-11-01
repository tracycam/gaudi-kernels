"""Explicit reduced-layer weight loading for diagnostic model runs only."""
import re

from gaudi_kernels.serving.module_bindings import ModuleBindings

_REGISTRATION = None
_LAYERS = None


def install(layers):
    global _REGISTRATION, _LAYERS
    if type(layers) is not int or not 1 <= layers <= 70:
        raise ValueError('Validation layer count must be 1..70')
    if _LAYERS is not None:
        if _LAYERS != layers:
            raise RuntimeError('Diagnostic layer count changed; use a fresh process')
        return _REGISTRATION.install() if _REGISTRATION is not None else None
    _LAYERS = layers
    if layers == 70:
        return None

    def bind(module):
        original = module.DefaultModelLoader.get_all_weights

        def filtered_weights(self, *args, **kwargs):
            for name, tensor in original(self, *args, **kwargs):
                match = re.search(r'(?:^|\.)layers\.(\d+)\.', name)
                if match is None or int(match[1]) < layers:
                    yield name, tensor

        module.DefaultModelLoader.get_all_weights = filtered_weights

    _REGISTRATION = ModuleBindings({
        'vllm.model_executor.model_loader.default_loader': bind,
    })
    return _REGISTRATION.install()
