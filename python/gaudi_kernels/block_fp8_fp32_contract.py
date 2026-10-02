from gaudi_kernels.engine.context import context as execution_context
"""Independent CPU reference and staged evidence for ordinary FP32 arithmetic.

This is not an emulation of an undisclosed GPU/MME reduction tree. Original
OCP encoding, native-half loss, FP32 dot order, and the declared epilogue are
separate contracts. No device result is used to construct the dot references.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np

CONTRACT = 'block_fp8_fp32_v1'
U32 = 2.0 ** -24
TINY32 = 2.0 ** -126


class UnsupportedArithmetic(ValueError):
    pass


def _decode(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    mag = codes & 127
    if np.any(mag == 127):
        raise ValueError('nonfinite original OCP E4M3FN code')
    exponent, mantissa = mag >> 3, mag & 7
    value = np.where(exponent == 0, mantissa.astype(np.float64) * 2.0 ** -9,
                     np.ldexp(1.0 + mantissa.astype(np.float64) / 8.0, exponent.astype(np.int32) - 7))
    return np.where(codes & 128, -value, value).astype(np.float32)


_POSITIVE = _decode(np.arange(127, dtype=np.uint8)).astype(np.float64)


def _encode(values):
    """Independent finite E4M3FN nearest-even search, including signed zero."""
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite quantization input')
    mag = np.minimum(np.abs(values).astype(np.float64), 448.0)
    hi = np.searchsorted(_POSITIVE, mag, side='left').clip(0, 126)
    lo = np.maximum(hi - 1, 0)
    dlo, dhi = mag - _POSITIVE[lo], _POSITIVE[hi] - mag
    choose_hi = (dhi < dlo) | ((dhi == dlo) & ((hi & 1) == 0))
    code = np.where(choose_hi, hi, lo).astype(np.uint8)
    return code | ((values.view(np.uint32) >> 24) & 128).astype(np.uint8)


def _round32(exact):
    exact = np.asarray(exact, dtype=np.float64)
    with np.errstate(over='ignore', under='ignore'):
        rounded = exact.astype(np.float32)
    # This version certifies normal-or-exact-zero operations. A nonzero exact
    # value rounded/FTZ'd to zero is explicitly outside the supported domain.
    if (not np.isfinite(exact).all() or not np.isfinite(rounded).all()
            or np.any((exact != 0) & (np.abs(exact) < TINY32))):
        raise UnsupportedArithmetic('FP32 overflow/subnormal/underflow operation; no gamma-only acceptance')
    return rounded


def _add(a, b):
    return _round32(np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64))


def _mul(a, b):
    return _round32(np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64))


def _balanced(values):
    work = np.array(values, dtype=np.float32, copy=True)
    while work.shape[-1] > 1:
        if work.shape[-1] & 1:
            work = np.concatenate((work, np.zeros((*work.shape[:-1], 1), dtype=np.float32)), axis=-1)
        work = _add(work[..., ::2], work[..., 1::2])
    return work[..., 0]


def _sequential(values):
    out = np.zeros(values.shape[:-1], dtype=np.float32)
    for index in range(values.shape[-1]):
        out = _add(out, values[..., index])
    return out


def _bf16(values):
    import torch
    result = torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(torch.bfloat16)
    if not torch.isfinite(result.float()).all():
        raise UnsupportedArithmetic('final BF16 overflow')
    return result


def _sha(tensor):
    import torch
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _gamma(n):
    return np.nextafter((n * U32) / (1.0 - n * U32), np.inf)


def _product_lattice(products):
    """Least dyadic unit; exactness of every possible tree when abs-sum fits.

    FP8 products are normal FP32 and exactly representable. Even the full
    128-product absolute sum fits FP64 exactly (the exponent span is bounded).
    """
    bits = np.abs(products).view(np.uint32)
    mantissa = (bits & 0x7fffff) | 0x800000
    lowbit = mantissa & (~mantissa + np.uint32(1))
    trailing = np.log2(lowbit).astype(np.int32)
    lsb = ((bits >> 23).astype(np.int32) - 127 - 23 + trailing)
    nonzero = products != 0
    least = np.where(nonzero, lsb, 1024).min(-1)
    least = np.where(nonzero.any(-1), least, 0)
    absolute = np.abs(products.astype(np.float64)).sum(-1)
    all_orders_exact = np.ldexp(absolute, -least) <= 2.0 ** 24
    bound = np.where(all_orders_exact, 0.0,
                     np.nextafter(_gamma(127) * np.nextafter(absolute, np.inf), np.inf))
    return least, absolute, all_orders_exact, bound


def _reject_source_subnormal(tensor, label):
    """Check source storage before a host DAZ/FTZ conversion can erase it."""
    import torch
    if tensor.dtype == torch.bfloat16:
        raw = tensor.contiguous().view(torch.int16).numpy().view(np.uint16)
        magnitude = raw & np.uint16(0x7fff)
        tiny = (magnitude != 0) & (magnitude < 0x0080)
    elif tensor.dtype == torch.float32:
        raw = tensor.contiguous().view(torch.int32).numpy().view(np.uint32)
        magnitude = raw & np.uint32(0x7fffffff)
        tiny = (magnitude != 0) & (magnitude < 0x00800000)
    else:
        raise ValueError('source-domain check requires BF16 or FP32')
    if tiny.any():
        raise UnsupportedArithmetic(label + ' source has nonzero FP32-subnormal value; host/device FTZ domain is unqualified')


def build_reference(x, weight, scales, *, bias=None, native_half=True):
    """M1 original-checkpoint CPU reference, preserving all block128 boundaries."""
    import torch
    if (x.device.type != 'cpu' or weight.device.type != 'cpu' or scales.device.type != 'cpu'
            or x.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn
            or scales.dtype != torch.float32 or x.ndim != 2 or weight.ndim != 2
            or x.shape[0] != 1 or x.shape[1] != weight.shape[1]):
        raise ValueError('CPU M1 BF16 X, original OCP E4M3FN [N,K] W and FP32 block scales required')
    n, k = weight.shape
    groups = (k + 127) // 128
    if n < 1 or k < 1 or tuple(scales.shape) != ((n + 127) // 128, groups):
        raise ValueError('block128 scale grid/shape mismatch')
    _reject_source_subnormal(x, 'BF16 activation')
    _reject_source_subnormal(scales, 'FP32 weight scale')
    sw = scales.contiguous().numpy().copy()
    if not np.isfinite(sw).all() or np.any(sw < 0) or not torch.isfinite(x.float()).all():
        raise ValueError('finite source values and nonnegative scales required')
    if bias is None:
        bias = torch.zeros(n, dtype=torch.float32)
    if bias.device.type != 'cpu' or bias.dtype != torch.float32 or tuple(bias.shape) != (n,):
        raise ValueError('CPU FP32 bias [N] required')
    _reject_source_subnormal(bias, 'FP32 bias')
    b = bias.contiguous().numpy().copy()
    _round32(b)
    activation = np.zeros((groups, 128), dtype=np.float32)
    activation.reshape(-1)[:k] = x.float().contiguous().numpy().reshape(-1)
    if np.any((activation != 0) & (np.abs(activation) < TINY32)):
        raise UnsupportedArithmetic('BF16 activation maps to FP32 subnormal; quant FTZ domain is unqualified')
    maximum = np.maximum(np.abs(activation).max(-1, keepdims=True), np.float32(1e-10))
    logical_sa = _mul(maximum, np.float32(1.0 / 448.0))
    quotient64 = activation.astype(np.float64) / logical_sa.astype(np.float64)
    if np.any((quotient64 != 0) & (np.abs(quotient64) < TINY32)):
        raise UnsupportedArithmetic('quantization division underflows/subnormal; FTZ domain is unqualified')
    # NumPy FP32 division is independent of TPC reciprocal/correction lowering.
    qa_ocp = _encode(np.clip(activation / logical_sa, -448, 448))
    source_codes = weight.contiguous().view(torch.uint8).numpy()
    histogram = np.bincount(source_codes.reshape(-1), minlength=256)
    if histogram[127] or histogram[255]:
        raise ValueError('nonfinite original OCP E4M3FN code')
    finite_codes = np.arange(256, dtype=np.uint8)
    finite_codes = finite_codes[(finite_codes & 127) != 127]
    half_table = np.zeros(256, dtype=np.uint8)
    half_table[finite_codes] = _encode(_decode(finite_codes) * np.float32(.5))
    qa = _encode(_decode(qa_ocp) * np.float32(.5)) if native_half else qa_ocp.copy()
    w = half_table[source_codes] if native_half else source_codes.copy()
    sa = _mul(logical_sa, np.float32(2)) if native_half else logical_sa.copy()
    sw = _mul(sw, np.float32(2)) if native_half else sw
    padded = np.zeros((n, groups * 128), dtype=np.uint8)
    padded[:, :k] = w
    prepared = padded.reshape(n, groups, 128).transpose(1, 0, 2).copy()
    shape = (groups, n)
    exact, absolute, bound = (np.empty(shape, dtype=np.float64) for _ in range(3))
    balanced, sequential = (np.empty(shape, dtype=np.float32) for _ in range(2))
    lattice = np.empty(shape, dtype=np.int32)
    all_exact = np.empty(shape, dtype=bool)
    for group in range(groups):
        products = _decode(prepared[group]) * _decode(qa[group])[None, :]
        exact[group] = products.astype(np.float64).sum(-1)
        lattice[group], absolute[group], all_exact[group], bound[group] = _product_lattice(products)
        balanced[group], sequential[group] = _balanced(products), _sequential(products)
        assert np.all(np.abs(balanced[group].astype(np.float64) - exact[group]) <= bound[group])
        assert np.all(np.abs(sequential[group].astype(np.float64) - exact[group]) <= bound[group])
    factors = _mul(sa, sw[np.arange(n) // 128].T)
    outputs = {}
    for name, partial, tree in (('balanced', balanced, _balanced), ('sequential', sequential, _sequential)):
        terms = _mul(partial, factors)
        outputs[name] = _bf16(_add(tree(terms.T), b)).reshape(1, n)
    tensor = lambda array: torch.from_numpy(np.array(array, copy=True))
    losses = np.zeros(256, dtype=np.float64)
    if native_half:
        losses[finite_codes] = _decode(half_table[finite_codes]).astype(np.float64) * 2 - _decode(finite_codes)
    half_activation = _decode(qa).astype(np.float64) * (2 if native_half else 1)
    return dict(contract=CONTRACT, encoding='native_half_rne_v1' if native_half else 'original_ocp',
                shape=[1, n, k], q_native=tensor(qa[:, None, :]), activation_scales=tensor(sa[:, None, :]),
                prepared_weight=tensor(prepared), prepared_scales=tensor(sw), bias=bias.clone(),
                partial_balanced=tensor(balanced[:, None, :]), partial_sequential=tensor(sequential[:, None, :]),
                exact_partial=exact, absolute_products=absolute, partial_bound=bound,
                lattice_exponent=lattice, all_orders_exact=all_exact, factors=factors,
                outputs=outputs, checkpoint_weight_sha256=_sha(weight), checkpoint_scales_sha256=_sha(scales),
                representation_loss=dict(
                    category='native_half_adaptation_not_fp32_accumulation' if native_half else 'none',
                    weight_changed_values=int(histogram[losses != 0].sum()),
                    weight_max_abs=float(np.max(np.abs(losses[histogram != 0]))),
                    activation_changed_values=int(np.count_nonzero(half_activation != _decode(qa_ocp))),
                    activation_max_abs=float(np.max(np.abs(half_activation - _decode(qa_ocp))))))


def reference(x, weight, scales, *, bias=None, native_half=True, tree='balanced'):
    if tree not in ('balanced', 'sequential'):
        raise ValueError('reference tree must be explicit balanced or sequential')
    return build_reference(x, weight, scales, bias=bias, native_half=native_half)['outputs'][tree]


_fmaf = None
_array_library = None
_array_fma = None
_array_identity = None


def require_fast_backend():
    """Production diagnostic gate: explicit frozen CPU library, never silent JIT."""
    global _array_library, _array_fma, _array_identity
    if _array_fma is not None:
        return _array_identity
    path = Path(execution_context().path('cpu_oracle', 'host')).resolve(strict=True)
    expected = execution_context().sha('cpu_oracle', 'host')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(expected) != 64 or actual != expected:
        raise ValueError('CPU FP32 oracle library SHA mismatch')
    build_path = Path(execution_context().path('cpu_oracle_build', 'host')).resolve(strict=True)
    build = json.loads(build_path.read_text())
    source_path = build_path.with_name('fp32_oracle.cpp')
    if (build.get('library_sha256') != actual
            or hashlib.sha256(source_path.read_bytes()).hexdigest() != build.get('source_sha256')
            or not {'-fno-fast-math', '-ffp-contract=off'}.issubset(build.get('command', []))
            or not build.get('runtime_libraries')):
        raise ValueError('missing/mismatched CPU oracle source/build/runtime provenance')
    for runtime_path, digest in build['runtime_libraries'].items():
        if hashlib.sha256(Path(runtime_path).read_bytes()).hexdigest() != digest:
            raise ValueError('CPU oracle runtime library changed: ' + runtime_path)
    library = ctypes.CDLL(str(path))
    library.gk_fp32_oracle_version.restype = ctypes.c_int
    if library.gk_fp32_oracle_version() != 1:
        raise ValueError('CPU FP32 oracle ABI version mismatch')
    function = library.gk_fp32_oracle_fma
    pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [pointer, pointer, pointer, pointer, ctypes.c_uint64]
    function.restype = ctypes.c_int
    _array_library, _array_fma = library, function
    _array_identity = dict(path=str(path), sha256=actual, version=1, cpu_only=True,
                           build_json=str(build_path), build_sha256=hashlib.sha256(build_path.read_bytes()).hexdigest(),
                           source_path=str(source_path), source_sha256=build['source_sha256'],
                           source_commit=build['source_commit'], runtime_libraries=build['runtime_libraries'])
    return _array_identity


def _fma(a, b, c):
    """C99 fmaf performs one correctly rounded FP32 operation, not a*b+c."""
    global _fmaf
    if _fmaf is None:
        _fmaf = ctypes.CDLL('libm.so.6').fmaf
        _fmaf.argtypes = [ctypes.c_float] * 3
        _fmaf.restype = ctypes.c_float
    aa, bb, cc = np.broadcast_arrays(a, b, c)
    # The range check is conservative for tiny cancellation. The actual result
    # is always obtained from fmaf, avoiding FP64-to-FP32 double rounding.
    _round32(aa.astype(np.float64) * bb.astype(np.float64) + cc.astype(np.float64))
    if _array_fma is None and execution_context().has('cpu_oracle.host'):
        require_fast_backend()
    if _array_fma is not None:
        arrays = [np.ascontiguousarray(v, dtype=np.float32) for v in (aa, bb, cc)]
        result = np.empty(aa.shape, dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        code = _array_fma(*(v.ctypes.data_as(pointer) for v in arrays), result.ctypes.data_as(pointer), result.size)
        if code:
            raise UnsupportedArithmetic('CPU FMA helper requires round-to-nearest environment')
    else:
        # Small CPU unit fixtures only. Full-model v1 validation explicitly
        # requires the array backend before workers begin quality diagnostics.
        values = [_fmaf(float(x), float(y), float(z)) for x, y, z in zip(aa.flat, bb.flat, cc.flat)]
        result = np.asarray(values, dtype=np.float32).reshape(aa.shape)
    _round32(result)
    return result


def epilogue_certificate(partial, factors, bias, reduction):
    """Independent declared epilogue order, with actual validated partials as inputs.

    This does not make those partials a dot oracle: they separately must pass
    the source-derived dyadic/lattice/error checks below.
    """
    values = np.asarray(partial, dtype=np.float32)
    total = np.zeros(values.shape[1], dtype=np.float32)
    correction = total.copy()
    if reduction not in ('sequential', 'neumaier_fp32', 'neumaier_isa_fp32'):
        raise ValueError('unknown epilogue contract')
    for value, factor in zip(values, factors):
        if reduction == 'sequential':
            total = _fma(value, factor, total)
        else:
            term = _mul(value, factor)
            product_error = _fma(value, factor, -term)
            updated = _add(total, term)
            first = _add(_add(total, -updated), term)
            second = _add(_add(term, -updated), total)
            residual = np.where(np.abs(total) >= np.abs(term), first, second)
            correction = _add(correction, _add(residual, product_error))
            total = updated
    if reduction != 'sequential':
        total = _add(total, correction)
    return _bf16(_add(total, np.asarray(bias, dtype=np.float32))).reshape(1, -1)


def audit(reference_data, actual, *, reduction, implementation=None):
    """Classify complete staged evidence; missing intermediates cannot pass."""
    import torch
    report = dict(contract=CONTRACT, encoding=reference_data['encoding'], reduction=reduction, passed=False,
                  numerical_passed=False, implementation_qualified=False,
                  classification='INCOMPLETE_EVIDENCE', stages={}, representation_loss=reference_data['representation_loss'],
                  checkpoint_weight_sha256=reference_data['checkpoint_weight_sha256'],
                  checkpoint_scales_sha256=reference_data['checkpoint_scales_sha256'])
    required = ('q_native', 'activation_scales', 'prepared_weight', 'prepared_scales', 'bias', 'partial', 'output')
    if set(required) - set(actual):
        report['missing'] = sorted(set(required) - set(actual))
        return report
    for name in required:
        if actual[name].device.type != 'cpu':
            raise ValueError('audit requires explicit CPU evidence, never implicit device readback')
    for name in required[:5]:
        expected, observed = reference_data[name], actual[name]
        if name in ('q_native', 'prepared_weight') and observed.dtype == torch.float8_e4m3fn:
            observed = observed.view(torch.uint8)
        same = expected.shape == observed.shape and expected.dtype == observed.dtype and torch.equal(expected, observed)
        # FP32 bit identity distinguishes signed zero as well as scale rounding.
        same = same and torch.equal(expected.contiguous().view(torch.uint8), observed.contiguous().view(torch.uint8))
        report['stages'][name] = dict(match=same, expected_sha256=_sha(expected), actual_sha256=_sha(observed))
        if not same:
            report.update(classification='REPRESENTATION_OR_INDEX_ERROR', failed_stage=name)
            return report
    partial = actual['partial']
    shape = reference_data['partial_balanced'].shape
    if partial.dtype != torch.float32 or partial.shape != shape or not torch.isfinite(partial).all():
        report.update(classification='PARTIAL_TYPE_SHAPE_OR_NONFINITE_ERROR')
        return report
    values = partial.numpy()[:, 0, :].astype(np.float64)
    units = np.ldexp(values, -reference_data['lattice_exponent'])
    error = np.abs(values - reference_data['exact_partial'])
    valid = (units == np.rint(units)) & (error <= reference_data['partial_bound'])
    worst = np.unravel_index(np.argmax(~valid) if not valid.all() else np.argmax(error), error.shape)
    report['stages']['partial'] = dict(passed=bool(valid.all()), actual_sha256=_sha(partial),
        max_abs_error=float(error.max()), all_orders_exact_cells=int(reference_data['all_orders_exact'].sum()),
        incompatible_cells=int((~valid).sum()), witness_group=int(worst[0]), witness_column=int(worst[1]),
        witness_error=float(error[worst]), witness_bound=float(reference_data['partial_bound'][worst]))
    if not valid.all():
        report.update(classification='FP32_DOT_CONTRACT_VIOLATION')
        return report
    try:
        expected = epilogue_certificate(values.astype(np.float32), reference_data['factors'],
                                         reference_data['bias'].numpy(), reduction)
    except UnsupportedArithmetic as exc:
        report.update(classification='UNSUPPORTED_ARITHMETIC_RANGE', reason=str(exc))
        return report
    output = actual['output']
    same = (output.dtype == torch.bfloat16 and output.shape == expected.shape
            and torch.equal(output.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)))
    report['stages']['epilogue'] = dict(match=same, expected_sha256=_sha(expected), actual_sha256=_sha(output))
    if not same:
        report.update(classification='DECLARED_FP32_EPILOGUE_MISMATCH')
        return report
    tree_matches = {name: torch.equal(output, value) for name, value in reference_data['outputs'].items()}
    from .fp32_artifact_binding import _VerifiedImplementation
    qualified = (isinstance(implementation, _VerifiedImplementation) and implementation.runtime_loaded
                 and implementation.reduction == reduction)
    report.update(passed=qualified, numerical_passed=True, implementation_qualified=qualified,
                  implementation=implementation.record if isinstance(implementation, _VerifiedImplementation) else None,
                  classification='STAGED_FP32_CONTRACT_COMPATIBLE' if qualified else 'IMPLEMENTATION_EVIDENCE_REQUIRED',
                  output_matches_independent_trees=tree_matches,
                  scope='bounded staged evidence; not proof of a particular GPU/MME tree or full-model quality')
    return report


def recipe_relation(plain, staged):
    """Bind an evidence-producing recipe to a separately completed plain output.

    This is an observed full-output relation, not proof of identical MME tiling
    or of the serving model's surrounding recipe. A difference can be legal
    FP32 tree variation; it leaves certification incomplete, not a math bug.
    """
    import torch
    if plain.device.type != 'cpu' or staged.device.type != 'cpu':
        raise ValueError('recipe relation requires explicit completed CPU readbacks')
    shape_dtype = plain.shape == staged.shape and plain.dtype == staged.dtype
    mismatches = None
    if shape_dtype:
        bits_a = plain.contiguous().view(torch.uint8)
        bits_b = staged.contiguous().view(torch.uint8)
        mismatches = int((bits_a != bits_b).sum())
    return dict(all_bits_equal=bool(shape_dtype and mismatches == 0),
                shape_dtype_match=shape_dtype, byte_mismatches=mismatches,
                plain_sha256=_sha(plain), staged_sha256=_sha(staged),
                relation='plain output completed before intermediate-retaining replay; plain output returned',
                scope='same-input output equivalence only; recipe identity/placement not inferred')
