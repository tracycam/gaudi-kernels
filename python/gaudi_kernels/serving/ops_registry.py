"""Standard, model-agnostic operator registration and lifecycle hooks for vLLM."""
import importlib
from gaudi_kernels.engine.context import ConfigError, context


def _active_context():
    try:
        return context()
    except ConfigError:
        return None


def register_general_operators(*, scope='qkv', block_fp8_policy='bf16_fp32',
                               norm_policy='vendor', expected_qkv_shape=None):
    """Explicitly register general-purpose operators before model construction."""
    runtime = _active_context()
    installed = {}
    if runtime is not None and runtime.has('norm'):
        import torch
        torch.ops.load_library(runtime.path('norm', 'torch'))
    try:
        from gaudi_kernels.vllm_norm import install_vllm_norm
        installed['norm'] = install_vllm_norm(norm_policy)
    except ImportError:
        installed['norm'] = {'installed': False}

    if runtime is not None and runtime.has('block_fp8'):
        hpu_fp8 = importlib.import_module('vllm_gaudi.ops.hpu_fp8')
        if runtime.has('reduce_isa'):
            from gaudi_kernels.serving.executor.qkv_neumaier_isa_runtime import prepare
            prepare()
        from gaudi_kernels.production_integration import install_block_fp8
        state = install_block_fp8(
            hpu_fp8,
            enabled=True,
            scope=scope,
            policy=block_fp8_policy,
            extension_path=runtime.path('block_fp8', 'torch'),
            expected_qkv_shape=expected_qkv_shape,
            keep_cpu_oracle=runtime.startup.keep_cpu_oracle,
        )
        installed['block_fp8'] = state.snapshot() if state is not None else {'installed': False}
    return installed


def prepare_model_operators(model, *, norm_policy=None, block_fp8_policy=None):
    """Prepare and activate general operators on any loaded model instance."""
    runtime = _active_context()
    if runtime is not None:
        engine = runtime.selection.engine
        if norm_policy is None:
            norm_policy = 'vendor' if engine.decode.norm.impl == 'vendor' else 'fp32'
        if block_fp8_policy is None:
            block_fp8_policy = runtime.selection.block_policy

    report = {}
    from gaudi_kernels import vllm_norm
    if vllm_norm._target is not None:
        prepared = vllm_norm.prepare_vllm_norm(model)
        policy_state = (vllm_norm.set_norm_policy(norm_policy if prepared['matched'] else 'vendor')
                        if norm_policy is not None else {'policy': vllm_norm.get_norm_policy()})
        report['norm'] = {'prepared': prepared, 'policy': policy_state}

    from gaudi_kernels import production_integration
    if production_integration._active is not None:
        if block_fp8_policy is not None:
            production_integration.set_policy(block_fp8_policy)
        report['block_fp8'] = production_integration.snapshot()
    return report
