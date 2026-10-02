"""Opt-in inference adapter for pinned vllm-gaudi HPURMSNorm (never Gemma).

Importing this module changes nothing. Policy defaults to vendor. The caller
must synchronize and invalidate/rebuild captured graphs when changing policy;
this module does not access devices, streams, graph caches, or A8 quantizers.
"""
import functools
import torch
from .residual_rmsnorm import pure_rmsnorm_bf16, residual_rmsnorm_bf16, reshape_bf16_graph

_policy = 'vendor'
_target = None
_original_init = None
_original_forward = None


def get_norm_policy():
    return _policy


def set_norm_policy(policy):
    """Set the Python dispatch choice at a caller-managed graph boundary.

    Existing HPU Graph recipes retain their captured operators. The deployment
    RPC must synchronize ranks and clear/rebuild those recipes before reuse.
    No policy automatically enables norm+A8 or changes a quantization contract.
    """
    global _policy
    if policy not in ('vendor', 'fp32'):
        raise ValueError('norm policy must be vendor or fp32')
    previous = _policy
    _policy = policy
    return {'previous': previous, 'policy': policy,
            'captured_graph_invalidation_required': previous != policy}


def _prepare_instance(module):
    weight = module.weight
    frozen = bool(weight.requires_grad)
    if frozen:
        weight.requires_grad_(False)
    # CustomOp.__init__ stores a bound forward method. Updating the class alone
    # does not redirect instances constructed before installation.
    bound = getattr(module, '_forward_method', None)
    fn = getattr(bound, '__func__', None)
    rebound = fn is _original_forward
    if rebound:
        module._forward_method = module.forward_oot
    dispatch_supported = rebound or fn is _forward
    return frozen, rebound, dispatch_supported


def prepare_vllm_norm(model):
    """Freeze loaded gamma once and rebind pre-existing OOT dispatch methods.

    Call after loading if a loader replaced Parameters. No tensor values move or
    change, and forward never detaches/clones/freezes a parameter.
    """
    if _target is None:
        raise RuntimeError('install_vllm_norm must run before preparing a model')
    report = {'matched': 0, 'gamma_frozen': 0, 'dispatch_rebound': 0,
              'non_oot_dispatch_retained': 0}
    for module in model.modules():
        # Explicit class identity excludes Gemma and unknown specializations.
        if type(module) is _target:
            frozen, rebound, supported = _prepare_instance(module)
            report['matched'] += 1
            report['gamma_frozen'] += int(frozen)
            report['dispatch_rebound'] += int(rebound)
            report['non_oot_dispatch_retained'] += int(not supported)
    return report


def _eligible(module, x, residual):
    if x.ndim not in (2, 3):
        return False
    weight = module.weight
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if x.device.type != 'hpu' or x.device != weight.device:
        return False
    if not x.is_contiguous() or not weight.is_contiguous() or x.requires_grad:
        return False
    h = x.shape[-1]
    if not (0 < h <= 8192 and 0 < x.numel() // h <= 2147483647
            and weight.ndim == 1 and weight.shape[0] == h
            and getattr(module, 'variance_size_override', None) is None):
        return False
    if residual is None:
        return True
    return (residual.ndim in (2, 3) and residual.dtype == torch.bfloat16
            and residual.device == x.device and residual.is_contiguous()
            and not residual.requires_grad and residual.shape[-1] == h
            and x.numel() == residual.numel())


def _pure_fp32(self, x):
    # Direct BF16 input, no synthetic residual or FP32 HBM expansion. Geometry
    # changes are graph-owned logical nodes, never temporary Torch views.
    shape = x.shape
    h = shape[-1]
    matrix = x if x.ndim == 2 else reshape_bf16_graph(x, (x.numel() // h, h))
    result = pure_rmsnorm_bf16(matrix, self.weight, self.variance_epsilon)
    return result if x.ndim == 2 else reshape_bf16_graph(result, shape)


def _vendor_shape(tensor, shape):
    # A no-op Torch view can become an orphan lazy input at a downstream public
    # CustomOp. Preserve the producer object or add a graph-owned logical edge.
    if tuple(tensor.shape) == tuple(shape):
        return tensor
    if (tensor.dtype == torch.bfloat16 and tensor.device.type == 'hpu'
            and tensor.is_contiguous() and not tensor.requires_grad):
        return reshape_bf16_graph(tensor, shape)
    return tensor.reshape(shape)


def _vendor_forward(self, x, residual):
    if residual is None:
        return _original_forward(self, x, residual)
    from vllm_gaudi.extension.kernels import rms_norm
    original_shape = x.shape
    # Keep the pinned vendor's BF16 add, square, gamma product and rsqrt path.
    # Only shape edges change; no normalized values are copied or retained.
    residual = residual + _vendor_shape(x, residual.shape)
    norm = rms_norm().apply(residual, self.weight, self.variance_epsilon)
    return _vendor_shape(norm, original_shape), residual


def _forward(self, x, residual=None):
    if _policy != 'fp32' or not _eligible(self, x, residual):
        return _vendor_forward(self, x, residual)
    if self.weight.requires_grad:
        raise RuntimeError('gamma was replaced after construction; call prepare_vllm_norm(model) after loading')
    if residual is None:
        return _pure_fp32(self, x)
    x_shape, r_shape = x.shape, residual.shape
    h = x_shape[-1]
    # The public CustomOp rejects a lazy no-op input view in some captures.
    # Preserve original 2D tensor objects; only actual 3D inputs need graph-owned logical flattening.
    matrix_x = x if x.ndim == 2 else reshape_bf16_graph(x, (x.numel() // h, h))
    matrix_r = residual if residual.ndim == 2 else reshape_bf16_graph(residual, (residual.numel() // h, h))
    result_r, result_y = residual_rmsnorm_bf16(
        matrix_x, matrix_r, self.weight, self.variance_epsilon)
    if x.ndim == 3:
        result_y = reshape_bf16_graph(result_y, x_shape)
    if residual.ndim == 3:
        result_r = reshape_bf16_graph(result_r, r_shape)
    # vLLM order is (normalized, residual), opposite the low-level core API.
    return result_y, result_r


def install_vllm_norm(policy='vendor', *, model=None):
    """Install before construction, or provide the existing resident model.

    Only HPURMSNorm is patched. Gemma, unsupported dtype/geometry,
    and noncontiguous inputs retain vendor arithmetic. Residual vendor shape
    restoration uses producer identity or a graph-owned BF16 logical reshape. Extension/kernel
    loading is explicit and remains the deployment's responsibility.
    """
    global _target, _original_init, _original_forward
    if policy not in ('vendor', 'fp32'):
        raise ValueError('norm policy must be vendor or fp32')
    from vllm_gaudi.ops.hpu_layernorm import HPURMSNorm
    if _target is None:
        _target = HPURMSNorm
        _original_init = HPURMSNorm.__init__
        _original_forward = HPURMSNorm.forward_oot

        @functools.wraps(_original_init)
        def init(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            # Inference-only preparation happens once, after gamma exists.
            if type(self) is _target:
                _prepare_instance(self)

        HPURMSNorm.__init__ = init
        HPURMSNorm.forward_oot = _forward
    elif _target is not HPURMSNorm:
        raise RuntimeError('HPURMSNorm class changed after installation')
    prepared = prepare_vllm_norm(model) if model is not None else None
    report = set_norm_policy(policy)
    report.update(installed=True, prepared=prepared,
                  scope='FP32 statistics/products for pure and residual RMS; BF16 residual/norm boundaries; no A8 fusion')
    return report
