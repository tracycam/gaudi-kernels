"""Explicit compatibility bindings for the pinned vLLM plugin.

Import has no registration/device side effects. The bootstrap calls install().
Model arithmetic stays above the bridge; reduced-model loading is test-only.
"""
from gaudi_kernels.engine.context import context as execution_context
from gaudi_kernels.serving.module_bindings import ModuleBindings

_REGISTRATION = None
_OWNER = None


def _apply(fullname, module):
    if fullname == 'vllm_gaudi.ops.hpu_mxfp4':
        from gaudi_kernels.serving.mxfp4_adapter import install_routing
        install_routing(module)
    if fullname == 'vllm.v1.executor.multiproc_executor':
        from gaudi_kernels.serving.host_placement import install_worker_spawn
        install_worker_spawn(module)
    if fullname == 'vllm_gaudi.ops.hpu_fp8' and execution_context().has('block_fp8'):
        if execution_context().has('reduce_isa'):
            from gaudi_kernels.serving.executor.qkv_neumaier_isa_runtime import prepare
            prepare()
        from gaudi_kernels.production_integration import install_block_fp8
        install_block_fp8(module, enabled=True, scope='all_block128', policy='bf16_fp32', extension_path=execution_context().path('block_fp8', 'torch'), keep_cpu_oracle=execution_context().startup.keep_cpu_oracle)
    if fullname == 'vllm_gaudi.ops.hpu_layernorm' and execution_context().has('norm'):
        import torch
        torch.ops.load_library(execution_context().path('norm', 'torch'))
        from gaudi_kernels.vllm_norm import install_vllm_norm
        install_vllm_norm('vendor')
        if execution_context().has('norm_grid24'):
            from gaudi_kernels.serving.executor.norm_grid24_runtime import prepare as prepare_grid24
            prepare_grid24()
    if fullname == 'vllm_gaudi.attention.backends.hpu_attn' and execution_context().has('swa'):
        import torch
        if execution_context().has('qkv_post'):
            torch.ops.load_library(execution_context().path('qkv_post', 'torch'))
        torch.ops.load_library(execution_context().path('swa', 'torch'))
        if execution_context().has('swa_av'):
            from gaudi_kernels.serving.executor.swa_av_runtime import prepare as prepare_swa_av
            prepare_swa_av()
        if execution_context().has('swa_batch'):
            from gaudi_kernels.serving.executor.swa_batch_runtime import prepare as prepare_swa_batch
            prepare_swa_batch()
        from gaudi_kernels.vllm_swa_install import install_vllm_swa
        install_vllm_swa('vendor')
    if fullname.endswith(('extension.ops', 'router.gate_linear', 'hpu_communicator')):
        from gaudi_kernels.serving.executor.precision_runtime import install as install_precision
        install_precision(module)
    if fullname.endswith('hpu_mxfp4'):
        from gaudi_kernels.serving.models.mimo_mxfp4 import install
        install(module)


def install():
    global _REGISTRATION, _OWNER
    runtime = execution_context()
    if _REGISTRATION is not None:
        if _OWNER is not runtime:
            raise RuntimeError('Operator context changed; use a fresh process')
        return _REGISTRATION.install()
    targets = [
        'vllm.v1.executor.multiproc_executor',
        'vllm_gaudi.distributed.device_communicators.hpu_communicator',
        'vllm_gaudi.extension.ops',
        'vllm.model_executor.layers.fused_moe.router.gate_linear',
        'vllm_gaudi.ops.hpu_mxfp4',
    ]
    for artifact, module in (
            ('block_fp8', 'vllm_gaudi.ops.hpu_fp8'),
            ('norm', 'vllm_gaudi.ops.hpu_layernorm'),
            ('swa', 'vllm_gaudi.attention.backends.hpu_attn')):
        if runtime.has(artifact):
            targets.append(module)
    _OWNER = runtime
    _REGISTRATION = ModuleBindings({
        name: (lambda module, name=name: _apply(name, module)) for name in targets
    })
    if runtime.startup.inventory:
        from pathlib import Path
        from gaudi_kernels.engine.inventory import start
        start(Path(__file__).resolve().parents[1])
    return _REGISTRATION.install()
