"""Explicit pre-load installation for the pinned Gaudi FP8 Linear lifecycle.

Importing this module does not import Torch/vLLM, load a library, or patch any
class. The executor calls install_block_fp8 AFTER hpu_fp8 finished importing,
BEFORE constructing model layers. Ordinary FP8 and unselected layers delegate
to the original Gaudi method. Policy switches require the runner to synchronize
and clear every existing graph cache first; this module never does either.
"""
from collections import Counter
from pathlib import Path
import hashlib
import math

_active = None
_CPU_POLICIES = ('cpu_ocp_a8_bf16_fp32', 'cpu_native_a8_bf16_fp32',
                 'cpu_ocp_a8_fp32_v1', 'cpu_native_a8_fp32_v1')
_POLICIES = ('bf16_fp32', 'decode_a8_bf16_fp32', 'small_batch_a8_bf16_fp32', *_CPU_POLICIES)


def _qkv(layer):
    return getattr(layer, 'prefix', '').rsplit('.', 1)[-1] == 'qkv_proj'


def _qkv_result(layer, value):
    """Record once at the actual projection producer, independent of consumer."""
    if _qkv(layer):
        from .serving.diagnostics import boundary_hashes as boundary
        boundary.emit(layer.prefix, 'qkv', value)
    return value


def _block128(method):
    config = method.quant_config
    return (getattr(method, 'block_quant', False)
            and getattr(config, 'is_checkpoint_fp8_serialized', False)
            and tuple(getattr(config, 'weight_block_size', ()) or ()) == (128, 128))


class BlockFP8Installation:
    """Process-local installation state; Python counts are not graph replays."""
    def __init__(self, scope, policy, expected_qkv_shape, extension_path, keep_cpu_oracle=False):
        if scope not in ('qkv', 'all_block128'):
            raise ValueError('scope must be qkv or explicit all_block128')
        if policy not in _POLICIES:
            raise ValueError(f'policy must be one of {_POLICIES}')
        if expected_qkv_shape is not None:
            expected_qkv_shape = tuple(expected_qkv_shape)
            if len(expected_qkv_shape) != 2 or any(not isinstance(n, int) or n <= 0 for n in expected_qkv_shape):
                raise ValueError('expected_qkv_shape must be positive (N,K), or explicit None')
        self.keep_cpu_oracle = keep_cpu_oracle
        self.scope = scope
        self.policy = policy
        self.expected_qkv_shape = expected_qkv_shape
        self.extension_path = extension_path
        self.extension_sha256 = None
        self.generation = 0
        self.selected_layers = {}
        self.python_apply_branch_counts = Counter()
        self.policy_history = []
        self.method_class = None
        self.audit_enabled = False
        self.audit_contract = 'legacy'
        self.audit_tag = None
        self.audit_frame = None
        self.audit_records = []
        self.reduction_policy = 'sequential'
        self.reduction_capture_counts = Counter()

    def set_reduction_policy(self, policy):
        if policy not in ('sequential', 'neumaier_fp32', 'neumaier_isa_fp32'):
            raise ValueError('Unknown block reduction policy')
        if policy == 'neumaier_fp32':
            import torch
            if not hasattr(torch.ops.gk_reduce_experiment, 'neumaier'):
                raise ValueError('Load the qualified Neumaier extension before capture')
        if policy == 'neumaier_isa_fp32':
            import torch
            if not hasattr(torch.ops.gk_reduce_isa, 'handschedule'):
                raise ValueError('Load the separately qualified ISA reducer before capture')
        self.reduction_policy = policy
        self.reduction_capture_counts.clear()

    def reducer(self):
        self.reduction_capture_counts[self.reduction_policy] += 1
        if self.reduction_policy == 'sequential':
            return None
        import torch
        if self.reduction_policy == 'neumaier_isa_fp32':
            return torch.ops.gk_reduce_isa.handschedule
        return torch.ops.gk_reduce_experiment.neumaier

    def choose(self, layer, rows):
        prepared = layer._gk_block_fp8
        # This is an explicit mixed-precision policy for the measured TP8 QKV
        # decode shape, not a universal crossover threshold or a device-value
        # branch. Other shapes preserve BF16 activation and FP32 scale math.
        # Opt-in verification policy preserves the same block-A8 arithmetic
        # across B*T<=16. It does not silently extend the partial-sum working
        # set into large-M prefill; that requires a separately admitted plan.
        a8_rows = 16 if self.policy == 'small_batch_a8_bf16_fp32' else 1
        if (self.policy in ('decode_a8_bf16_fp32', 'small_batch_a8_bf16_fp32')
                and _qkv(layer) and 1 <= rows <= a8_rows
                and (prepared.n, prepared.k) == (3392, 6144)):
            return 'per_block_fp8', 'fp32'
        return 'bf16', 'fp32'

    def set_policy(self, policy):
        """Call ONLY after runner synchronization and all graph-cache clears.

        This changes Python capture/eager dispatch, not existing device recipes.
        Keeping a previous HPU Graph would replay its previous policy. No device
        operation is performed or inferred here; quiescence is the RPC contract.
        """
        if policy not in _POLICIES:
            raise ValueError(f'policy must be one of {_POLICIES}')
        if policy in _CPU_POLICIES:
            import os
            if not self.keep_cpu_oracle:
                raise ValueError('CPU diagnostic policy requires original host weights retained at load time')
        if policy != self.policy:
            self.policy_history.append({'generation': self.generation, 'policy': self.policy,
                'python_apply_branch_counts': dict(self.python_apply_branch_counts)})
            self.generation += 1
            self.policy = policy
            self.python_apply_branch_counts.clear()
        return self.snapshot()

    def snapshot(self):
        return {'installed': True, 'scope': self.scope, 'policy': self.policy,
                'generation': self.generation, 'expected_qkv_shape': self.expected_qkv_shape,
                'extension_path': self.extension_path,
                'extension_sha256': self.extension_sha256,
                'reduction_policy': self.reduction_policy,
                'reduction_capture_counts': dict(self.reduction_capture_counts),
                'reduction_operator': {'sequential': 'gaudi_block_fp8::reduce',
                    'neumaier_fp32': 'gk_reduce_experiment::neumaier',
                    'neumaier_isa_fp32': 'gk_reduce_isa::handschedule'}[self.reduction_policy],
                'selected_layer_count': len(self.selected_layers),
                'selected_layers': dict(self.selected_layers),
                'python_apply_branch_counts': dict(self.python_apply_branch_counts),
                'count_scope': 'Python apply calls, including capture/warmup/eager; excludes cached graph replays',
                'policy_history': list(self.policy_history)}


def install_block_fp8(hpu_fp8_module, *, enabled=True, scope='qkv', policy='bf16_fp32',
                     extension_path=None, expected_qkv_shape=(3392, 6144), keep_cpu_oracle=False):
    """Install after plugin import and before model construction/weight loading.

    Set GC_KERNEL_PATH to the block TPC database before any Habana import.
    extension_path loads the separate public bridge once; None requires that
    its operators were already loaded. Default scope is QKV only, with the
    measured MiMo TP8 local [3392,6144]/[27,48] shape checked before preparation.
    Passing expected_qkv_shape=None explicitly admits other QKV dimensions to
    the BF16 reference route; it does not qualify their performance or TP layout.
    """
    global _active
    if not enabled:
        if getattr(hpu_fp8_module, '_gk_block_fp8_installation', None) is not None:
            raise RuntimeError('cannot disable an installed loader in a live process')
        return None
    if hpu_fp8_module.__name__ != 'vllm_gaudi.ops.hpu_fp8':
        raise ValueError('install on the completed vllm_gaudi.ops.hpu_fp8 module')
    resolved = str(Path(extension_path).resolve(strict=True)) if extension_path is not None else None
    state = BlockFP8Installation(scope, policy, expected_qkv_shape, resolved, keep_cpu_oracle)
    existing = getattr(hpu_fp8_module, '_gk_block_fp8_installation', None)
    if existing is not None:
        if (existing.scope, existing.policy, existing.expected_qkv_shape, existing.extension_path) != (
                state.scope, state.policy, state.expected_qkv_shape, state.extension_path):
            raise RuntimeError('installation differs; use set_policy only after runner quiescence')
        if (hpu_fp8_module.Fp8LinearMethod is not existing.method_class
                or hpu_fp8_module.fp8.Fp8LinearMethod is not existing.method_class):
            raise RuntimeError('another hook replaced a registered FP8 class after installation')
        _active = existing
        return existing
    original = hpu_fp8_module.Fp8LinearMethod
    upstream = hpu_fp8_module.OrigFp8LinearMethod
    if (hpu_fp8_module.fp8.Fp8LinearMethod is not original
            or not issubclass(original, upstream)):
        raise RuntimeError('unexpected Gaudi/upstream FP8 aliases; install before competing loader hooks')
    import torch
    required = ('quant', 'batch_mm', 'reduce', 'decode', 'decode_fast', 'mm', 'finish', 'reshape')
    registered = all(hasattr(torch.ops.gaudi_block_fp8, name) for name in required)
    if registered and resolved is not None and resolved not in torch.ops.loaded_libraries:
        raise RuntimeError('block operators already registered by another library path; pass extension_path=None')
    if resolved is not None:
        state.extension_sha256 = hashlib.sha256(Path(resolved).read_bytes()).hexdigest()
        torch.ops.load_library(resolved)
    if any(not hasattr(torch.ops.gaudi_block_fp8, name) for name in required):
        raise RuntimeError('load gaudi_block_fp8_torch.so before preparing model weights')
    from .vllm_block_fp8 import make_vllm_block_fp8_method
    from .block_fp8 import linear_block_fp8, linear_block_fp8_quantized
    reference_method = make_vllm_block_fp8_method(activation='bf16', scale_math='fp32', keep_cpu_oracle=keep_cpu_oracle)

    class ProductionGaudiFp8LinearMethod(original):
        def create_weights(self, layer, *args, **kwargs):
            selected = _block128(self) and (scope == 'all_block128' or _qkv(layer))
            if not selected:
                return original.create_weights(self, layer, *args, **kwargs)
            if hasattr(layer, '_gk_block_fp8_selected'):
                raise ValueError('selected layer create_weights called twice')
            layer._gk_block_fp8_selected = True
            # Direct upstream loader: never the Gaudi half wrapper, channel
            # conversion, or BF16/FP32 expanded persistent weight cache.
            return reference_method.create_weights(self, layer, *args, **kwargs)

        def process_weights_after_loading(self, layer):
            if not getattr(layer, '_gk_block_fp8_selected', False):
                return original.process_weights_after_loading(self, layer)
            if (_qkv(layer) and state.expected_qkv_shape is not None
                    and tuple(layer.weight.shape) != state.expected_qkv_shape):
                raise ValueError(f'QKV local weight {tuple(layer.weight.shape)} differs from explicit expected {state.expected_qkv_shape}')
            if (state.expected_qkv_shape == (3392, 6144) and _qkv(layer)
                    and getattr(layer, 'tp_size', 8) != 8):
                raise ValueError('the MiMo original row-group contract requires TP8')
            layer.quant_config = self.quant_config
            reference_method.process_weights_after_loading(self, layer)
            prepared = layer._gk_block_fp8
            if not prepared.bf16_fp32_scale_safe:
                raise ValueError('selected production layer cannot run the finite BF16/FP32 reference policy')
            # The source factory's fixed policy label does not describe this
            # installation's explicit mutable capture policy.
            del layer._gk_block_fp8_policy
            layer._gk_block_fp8_policy_owner = state
            prefix = getattr(layer, 'prefix', '<unknown>')
            state.selected_layers[prefix] = {'shape': [prepared.n, prepared.k],
                'scales_shape': list(prepared.scales.shape), 'encoding': prepared.encoding,
                'checkpoint_weight_sha256': prepared.checkpoint_weight_sha256,
                'checkpoint_scales_sha256': prepared.checkpoint_scales_sha256,
                'rounded_half_values': prepared.rounded_half_values}

        def apply(self, layer, x, bias=None):
            if not getattr(layer, '_gk_block_fp8_selected', False):
                return original.apply(self, layer, x, bias)
            prepared = layer._gk_block_fp8
            if x.ndim < 2 or x.shape[-1] != prepared.k or not x.is_contiguous():
                raise ValueError('selected block FP8 requires contiguous [...,K] input with the loaded K')
            rows = math.prod(x.shape[:-1])
            if state.policy in _CPU_POLICIES and rows == 1:
                if not hasattr(layer, '_gk_oracle_weight'):
                    raise RuntimeError('CPU diagnostic was not enabled before loading this layer')
                if state.policy.endswith('_fp32_v1'):
                    from .block_fp8_fp32_contract import reference
                    arithmetic = 'block_fp8_fp32_v1_balanced_dot_separate_mul_balanced_sum'
                else:
                    from .block_fp8_oracle import reference
                    arithmetic = 'fp64_reference'
                state.python_apply_branch_counts[state.policy + '/' + arithmetic] += 1
                result = reference(x.detach().cpu().reshape(1, prepared.k),
                    layer._gk_oracle_weight, layer._gk_oracle_scales,
                    native_half=state.policy.startswith('cpu_native_'),
                    bias=None if bias is None else bias.detach().cpu())
                return _qkv_result(layer,result.reshape(*x.shape[:-1], prepared.n).to(x.device))
            activation, scale_math = state.choose(layer, rows)
            state.python_apply_branch_counts[f'{activation}/{scale_math}'] += 1
            # A graph-owned logical node preserves producer edges. Temporary
            # Torch views of vendor norm outputs failed at capture_end.
            flat = x if x.ndim == 2 else torch.ops.gaudi_block_fp8.reshape(x, [rows, prepared.k])
            staged_audit = (state.audit_enabled and state.audit_contract == 'fp32_arithmetic_v1'
                            and state.audit_frame is not None and activation == 'per_block_fp8' and rows == 1)
            # Complete the ordinary output first, without retaining intermediate
            # tensors as extra graph outputs. The staged copy is diagnostic only.
            staged = {} if staged_audit else None
            reducer = state.reducer() if activation == 'per_block_fp8' else None
            y = linear_block_fp8(flat, prepared, activation=activation,
                                 scale_math=scale_math, bias=bias,
                                 reduce_op=reducer)
            if staged_audit:
                import time
                from .block_fp8_fp32_contract import build_reference, audit, UnsupportedArithmetic, _sha
                from .fp32_artifact_binding import verify_artifacts
                start = time.monotonic()
                if not hasattr(layer, '_gk_oracle_weight'):
                    raise RuntimeError('staged audit requires original host checkpoint weights')
                plain_cpu = y.detach().cpu().contiguous()
                # y has executed before q/P are kept alive. No serving path
                # receives the evidence-producing copy of the computation.
                staged_y = linear_block_fp8(flat, prepared, activation=activation,
                    scale_math=scale_math, bias=bias, reduce_op=reducer, audit_tensors=staged)
                staged_cpu = staged_y.detach().cpu().contiguous()
                from .block_fp8_fp32_contract import recipe_relation
                relation = recipe_relation(plain_cpu, staged_cpu)
                x_cpu = x.detach().cpu().reshape(1, prepared.k)
                actual = {name: tensor.detach().cpu().contiguous() for name, tensor in staged.items()}
                try:
                    expected = build_reference(x_cpu, layer._gk_oracle_weight, layer._gk_oracle_scales,
                        bias=None if bias is None else bias.detach().cpu(), native_half=True)
                    implementation = verify_artifacts(state.reduction_policy)
                    checked = audit(expected, actual, reduction=state.reduction_policy, implementation=implementation)
                    if not relation['all_bits_equal'] and checked.get('numerical_passed'):
                        checked.update(passed=False, classification='AUDIT_RECIPE_VARIATION_INCOMPLETE')
                except UnsupportedArithmetic as exc:
                    checked = {'contract': 'block_fp8_fp32_v1', 'passed': False,
                               'classification': 'UNSUPPORTED_ARITHMETIC_RANGE', 'reason': str(exc)}
                state.audit_records.append({'layer': layer.prefix, 'policy': state.policy, 'same_input': True,
                    'quality_contract': state.audit_contract, 'tag': state.audit_tag,
                    'frame': dict(state.audit_frame), 'input_sha256': _sha(x_cpu),
                    'reduction_policy': state.reduction_policy, 'plain_vs_staged': relation,
                    'source_layout': getattr(layer, '_gk_oracle_layout', None),
                    'rounded_half_values': prepared.rounded_half_values,
                    'checkpoint_weight_sha256': prepared.checkpoint_weight_sha256,
                    'checkpoint_scales_sha256': prepared.checkpoint_scales_sha256,
                    'fp32_contract': checked, 'diagnostic_wall_s': time.monotonic() - start,
                    'scope': 'graph-disabled same-input diagnostic including readbacks; excluded from TPS'})
            elif state.audit_enabled and state.audit_contract == 'legacy' and activation == 'per_block_fp8' and rows == 1:
                from .block_fp8_oracle import reference
                if not hasattr(layer, '_gk_oracle_weight'):
                    raise RuntimeError('same-input audit requires original host checkpoint weights')
                x_cpu = x.detach().cpu().reshape(1, prepared.k)
                actual = y.detach().cpu().double()
                comparisons = {}
                for native_half in (False, True):
                    expected = reference(x_cpu, layer._gk_oracle_weight, layer._gk_oracle_scales,
                        native_half=native_half, bias=None if bias is None else bias.detach().cpu()).double()
                    difference = actual - expected
                    comparisons['native_half' if native_half else 'original_ocp'] = {
                        'relative_l2': float(difference.norm() / expected.norm().clamp_min(1e-30)),
                        'max_abs': float(difference.abs().max()),
                        'bf16_value_mismatches': int((actual != expected).sum()),
                        'finite': bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())}
                state.audit_records.append({'layer': layer.prefix, 'policy': state.policy,
                    'same_input': True, 'comparisons': comparisons})
            _qkv_result(layer,y)
            return y if x.ndim == 2 else torch.ops.gaudi_block_fp8.reshape(y, [*x.shape[:-1], prepared.n])

        def apply_quantized(self, layer, q, scales, bias=None):
            """Norm-fused input consumer; returns [M,N], with no second quant."""
            if not getattr(layer, '_gk_block_fp8_selected', False):
                raise ValueError('quantized consumer requires a selected block128 layer')
            activation, scale_math = state.choose(layer, q.shape[1])
            if activation != 'per_block_fp8':
                raise ValueError('current policy requires BF16 activation, not prequantized input')
            state.python_apply_branch_counts['external_block_fp8/fp32'] += 1
            return _qkv_result(layer,linear_block_fp8_quantized(q, scales, layer._gk_block_fp8, bias=bias,
                                             reduce_op=state.reducer()))

        def dequant_fp8_weight(self, layer):
            if getattr(layer, '_gk_block_fp8_selected', False):
                raise RuntimeError('selected block FP8 storage cannot be replaced with an expanded weight cache')
            return original.dequant_fp8_weight(self, layer)

    # vLLM selects weight_loader_v2 by this exact class-name string.
    ProductionGaudiFp8LinearMethod.__name__ = original.__name__
    ProductionGaudiFp8LinearMethod.__qualname__ = original.__qualname__
    ProductionGaudiFp8LinearMethod.__module__ = hpu_fp8_module.__name__
    state.method_class = ProductionGaudiFp8LinearMethod
    hpu_fp8_module.Fp8LinearMethod = ProductionGaudiFp8LinearMethod
    hpu_fp8_module.fp8.Fp8LinearMethod = ProductionGaudiFp8LinearMethod
    hpu_fp8_module._gk_block_fp8_installation = state
    _active = state
    return state


def set_policy(policy):
    """Runner RPC hook: synchronize and clear all graph caches BEFORE calling."""
    if _active is None:
        raise RuntimeError('block FP8 production integration is not installed in this worker')
    return _active.set_policy(policy)


def snapshot():
    """JSON-compatible worker status; no HPU queries or synchronization."""
    return {'installed': False} if _active is None else _active.snapshot()


def set_reduction_policy(policy):
    """Runner must drain native replay and clear graphs before every call."""
    if _active is None:
        raise RuntimeError('block FP8 production integration is not installed')
    _active.set_reduction_policy(policy)
    return _active.snapshot()


def configure_same_input_audit(enabled, *, contract='legacy', tag=None):
    """Diagnostic only: caller disables native replay/HPU Graphs before arming."""
    if _active is None:
        raise RuntimeError('production integration not installed')
    if contract not in ('legacy', 'fp32_arithmetic_v1'):
        raise ValueError('unknown same-input audit contract')
    previous = list(_active.audit_records)
    _active.audit_records.clear()
    _active.audit_enabled = enabled
    _active.audit_contract = contract
    _active.audit_tag = tag
    _active.audit_frame = None
    return previous


def set_same_input_audit_frame(frame):
    """Quality-only runner binds one actual decode token/position to all QKV rows."""
    if _active is None:
        raise RuntimeError('production integration not installed')
    _active.audit_frame = None if frame is None else dict(frame)
