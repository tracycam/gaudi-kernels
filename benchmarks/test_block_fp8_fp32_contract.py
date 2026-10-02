"""CPU mathematical/representation gates; not a device or full-model gate."""
import numpy as np
import pytest
import torch

from gaudi_kernels.block_fp8 import prepare_block_fp8
from gaudi_kernels.block_fp8_fp32_contract import (
    UnsupportedArithmetic, _balanced, _decode, _encode, _round32, _sequential,
    audit, build_reference, epilogue_certificate)


def fixture(n=9, k=257):
    generator = torch.Generator().manual_seed(928128)
    x = torch.randn(1, k, generator=generator).bfloat16()
    raw = torch.randint(0, 127, (n, k), generator=generator, dtype=torch.uint8)
    raw |= torch.randint(0, 2, (n, k), generator=generator, dtype=torch.uint8) * 128
    w = raw.view(torch.float8_e4m3fn)
    s = torch.rand((n + 127) // 128, (k + 127) // 128, generator=generator) * .01 + .001
    return x, w, s


def actual_from(ref, reduction='sequential', partial=None):
    actual = {k: ref[k].clone() for k in ('q_native', 'activation_scales', 'prepared_weight', 'prepared_scales', 'bias')}
    actual['partial'] = ref['partial_balanced'].clone() if partial is None else partial
    actual['output'] = epilogue_certificate(actual['partial'].numpy()[:, 0, :], ref['factors'],
                                            ref['bias'].numpy(), reduction)
    return actual


def test_all_finite_ocp_codes_decode_encode_and_half_adaptation():
    codes = np.arange(256, dtype=np.uint8)
    codes = codes[(codes & 127) != 127]
    values = torch.from_numpy(codes).view(torch.float8_e4m3fn).float()
    assert np.array_equal(_decode(codes).view(np.uint32), values.numpy().view(np.uint32))
    assert np.array_equal(_encode(values.numpy()), codes)
    half = (values * .5).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    assert np.array_equal(_encode(values.numpy() * .5), half)


def test_quantization_and_prepared_layout_match_separate_existing_cpu_implementation():
    x, w, scales = fixture(n=129)
    ref = build_reference(x, w, scales)
    prepared = prepare_block_fp8(w, scales)
    assert torch.equal(ref['prepared_weight'], prepared.weight.view(torch.uint8))
    assert torch.equal(ref['prepared_scales'], prepared.scales)
    padded = torch.zeros(3, 128); padded.flatten()[:257] = x.float().flatten()
    sa = padded.abs().amax(-1, keepdim=True).clamp_min(1e-10) * (1 / 448)
    q = (padded / sa).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    q = (q * .5).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(ref['q_native'][:, 0], q)
    assert torch.equal(ref['activation_scales'][:, 0].view(torch.int32), (sa * 2).view(torch.int32))
    assert ref['representation_loss']['weight_changed_values'] == prepared.rounded_half_values


@pytest.mark.parametrize('reduction', ['sequential', 'neumaier_fp32', 'neumaier_isa_fp32'])
def test_staged_reference_and_explicit_epilogue_certificate(reduction):
    ref = build_reference(*fixture())
    actual = actual_from(ref, reduction)
    report = audit(ref, actual, reduction=reduction)
    assert report['numerical_passed'] and report['stages']['partial']['passed']
    assert not report['passed'] and report['classification'] == 'IMPLEMENTATION_EVIDENCE_REQUIRED'
    assert report['representation_loss']['category'] == 'native_half_adaptation_not_fp32_accumulation'


@pytest.mark.parametrize('stage', ['q_native', 'activation_scales', 'prepared_weight', 'prepared_scales', 'bias'])
def test_single_bit_representation_change_cannot_hide_inside_gamma(stage):
    ref = build_reference(*fixture())
    actual = actual_from(ref)
    actual[stage].view(torch.uint8).reshape(-1)[0] ^= 1
    report = audit(ref, actual, reduction='sequential')
    assert not report['passed'] and report['failed_stage'] == stage


def test_wrong_row_layout_and_missing_partial_fail_before_output_accuracy():
    ref = build_reference(*fixture())
    actual = actual_from(ref)
    actual['prepared_weight'] = actual['prepared_weight'].roll(1, dims=1)
    assert audit(ref, actual, reduction='sequential')['classification'] == 'REPRESENTATION_OR_INDEX_ERROR'
    del actual['partial']
    assert audit(ref, actual, reduction='sequential')['classification'] == 'INCOMPLETE_EVIDENCE'


def test_all_orders_exact_zero_bound_rejects_one_ulp_corruption():
    x = torch.ones(1, 128, dtype=torch.bfloat16)
    w = torch.ones(2, 128).to(torch.float8_e4m3fn)
    ref = build_reference(x, w, torch.ones(1, 1))
    assert ref['all_orders_exact'].all() and not ref['partial_bound'].any()
    actual = actual_from(ref)
    actual['partial'][0, 0, 0] = torch.nextafter(actual['partial'][0, 0, 0], torch.tensor(float('inf')))
    assert audit(ref, actual, reduction='sequential')['classification'] == 'FP32_DOT_CONTRACT_VIOLATION'


def test_illegal_bf16_partial_rounding_is_rejected():
    ref = build_reference(*fixture(n=129, k=128))
    actual = actual_from(ref)
    actual['partial'] = actual['partial'].bfloat16().float()
    report = audit(ref, actual, reduction='sequential')
    assert not report['passed'] and report['classification'] == 'FP32_DOT_CONTRACT_VIOLATION'


def test_exact_underflow_and_overflow_are_unsupported_not_accepted_as_zero():
    for value in (2.0 ** -140, 2.0 ** -160, 2.0 ** 128):
        with pytest.raises(UnsupportedArithmetic):
            _round32(value)
    assert _round32(0).item() == 0


def test_two_tree_outputs_are_not_an_acceptance_interval():
    values = np.array([2 ** 24, 1, -2 ** 24, 1], dtype=np.float32)
    assert _sequential(values) == 1 and _balanced(values) == 1
    assert _balanced(values[[0, 2, 1, 3]]) == 2


def test_native_half_loss_is_not_labeled_as_fp32_accumulation():
    x = torch.zeros(1, 128, dtype=torch.bfloat16); x[0, 0] = 1
    w = torch.zeros(1, 128); w[0, 0] = 1 / 512
    w = w.to(torch.float8_e4m3fn)
    native = build_reference(x, w, torch.ones(1, 1))
    original = build_reference(x, w, torch.ones(1, 1), native_half=False)
    assert native['outputs']['balanced'].item() == 0
    assert original['outputs']['balanced'].item() == 1 / 512
    assert native['representation_loss']['weight_changed_values'] == 1
    assert original['representation_loss']['weight_changed_values'] == 0


@pytest.mark.parametrize('row', [3072, 3328])
def test_tp_qkv_scale_boundary_swap_is_not_a_dot_order_error(row):
    x, w, scales = fixture(n=3392, k=129)
    scales[:, 0] = torch.arange(1, 28).float()
    ref = build_reference(x, w, scales)
    actual = actual_from(ref)
    group = row // 128
    actual['prepared_scales'][[group - 1, group]] = actual['prepared_scales'][[group, group - 1]]
    report = audit(ref, actual, reduction='sequential')
    assert not report['passed'] and report['failed_stage'] == 'prepared_scales'


def test_recipe_relation_is_complete_raw_bits_not_float_equality():
    from gaudi_kernels.block_fp8_fp32_contract import recipe_relation
    plain = torch.tensor([0., -0., 1.], dtype=torch.bfloat16)
    assert recipe_relation(plain, plain.clone())['all_bits_equal']
    staged = plain.clone(); staged[1] = 0.
    assert torch.equal(plain, staged)
    checked = recipe_relation(plain, staged)
    assert not checked['all_bits_equal'] and checked['byte_mismatches'] == 1
    assert not recipe_relation(plain, plain.float())['all_bits_equal']


def test_unknown_implementation_library_cannot_qualify_numeric_pass(tmp_path, monkeypatch):
    from gaudi_kernels.fp32_artifact_binding import verify_artifacts
    library = tmp_path / 'unknown.so'; library.write_bytes(b'not a qualified implementation')
    monkeypatch.setenv('GK_BLOCK_FP8_TPC_LIBRARY', str(library))
    with pytest.raises(ValueError, match='unknown FP32 implementation bytes'):
        verify_artifacts('sequential', require_loaded=False)


@pytest.mark.parametrize('negative', [False, True])
def test_bf16_subnormal_input_is_unsupported_not_a_representation_bug(negative):
    x, w, scales = fixture(n=1, k=128)
    x[0, 0] = (-1 if negative else 1) * 2.0 ** -133
    with pytest.raises(UnsupportedArithmetic, match='activation source has nonzero FP32-subnormal'):
        build_reference(x, w, scales)


def test_normal_activation_with_extreme_dynamic_range_rejects_quant_underflow():
    x, w, scales = fixture(n=1, k=128)
    x[0, 0] = 2.0 ** -126; x[0, 1] = 2.0 ** 127
    with pytest.raises(UnsupportedArithmetic, match='quantization division'):
        build_reference(x, w, scales)


@pytest.mark.parametrize('flush', [False, True])
@pytest.mark.parametrize('source', ['activation', 'scale', 'bias'])
def test_source_subnormal_rejection_survives_host_daz_ftz(flush, source):
    x, w, scales = fixture(n=1, k=128)
    bias = torch.zeros(1, dtype=torch.float32)
    # Integer storage construction avoids a floating conversion before the
    # tested check. Positive and negative subnormals are both unsupported.
    if source == 'activation':x.view(torch.int16)[0, 0] = -32767
    if source == 'scale':scales.view(torch.int32)[0, 0] = 1
    if source == 'bias':bias.view(torch.int32)[0] = -2147483647
    try:
        torch.set_flush_denormal(flush)
        with pytest.raises(UnsupportedArithmetic, match='source has nonzero FP32-subnormal'):
            build_reference(x, w, scales, bias=bias)
    finally:
        torch.set_flush_denormal(False)
