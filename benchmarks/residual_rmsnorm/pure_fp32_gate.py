"""Real HPURMSNorm pure/residual FP32 -> production QKV capture boundary.

This gate separately checks norm FP32 values/rounding, the compiled graph's
precision, and the same-norm-input QKV producer boundary. It does not certify
the block MME arithmetic or substitute a CPU tree for a GPU implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

p = argparse.ArgumentParser()
p.add_argument('--norm-extension', required=True)
p.add_argument('--block-extension', required=True)
a = p.parse_args()
out = Path(os.environ['PROBE_OUT'])
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',
                 DUMP_POST_GRAPHS=str(out / 'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from vllm.config import VllmConfig, CompilationConfig, set_current_vllm_config
from vllm_gaudi.ops.hpu_layernorm import HPURMSNorm
from vllm_gaudi.extension.kernels import rms_norm

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'python'))
from gaudi_kernels import vllm_norm
from gaudi_kernels.block_fp8 import prepare_block_fp8
from gaudi_kernels.production_integration import install_block_fp8

torch.set_num_threads(4)
torch.manual_seed(280932)
torch.ops.load_library(str(Path(a.norm_extension).resolve()))


def sync():
    hc.mark_step()
    torch.hpu.synchronize()


def bits_equal(x, y):
    return (x.dtype == y.dtype and x.shape == y.shape
            and torch.equal(x.contiguous().view(torch.uint8),
                            y.contiguous().view(torch.uint8)))


def relative(x, y):
    return float((x.double() - y.double()).norm() / y.double().norm().clamp_min(1e-30))


config = VllmConfig(compilation_config=CompilationConfig(custom_ops=['all']))
with set_current_vllm_config(config):
    resident = HPURMSNorm(6144, dtype=torch.bfloat16)
    installation = vllm_norm.install_vllm_norm('fp32', model=torch.nn.Sequential(resident))
assert installation['prepared']['matched'] == 1 and not resident.weight.requires_grad
gamma0 = (torch.randn(6144) * .2 + 1).bfloat16()
resident.weight.copy_(gamma0)
resident = resident.to('hpu')
sync()


class Original:
    def __init__(self, config):
        self.quant_config = config
        self.block_quant = True


class Fp8LinearMethod(Original):
    def create_weights(self, *args, **kwargs):
        raise AssertionError('fixture is prepared explicitly')

    def process_weights_after_loading(self, *args):
        raise AssertionError('fixture is prepared explicitly')

    def apply(self, *args):
        raise AssertionError('must dispatch production apply')


plugin = types.ModuleType('vllm_gaudi.ops.hpu_fp8')
plugin.OrigFp8LinearMethod = Original
plugin.Fp8LinearMethod = Fp8LinearMethod
plugin.fp8 = types.SimpleNamespace(Fp8LinearMethod=Fp8LinearMethod)
sys.modules[plugin.__name__] = plugin
state = install_block_fp8(plugin, extension_path=a.block_extension)
method = plugin.Fp8LinearMethod(types.SimpleNamespace(
    is_checkpoint_fp8_serialized=True, weight_block_size=(128, 128)))
n, k = 3392, 6144
codes = torch.randint(16, 96, (n, k), dtype=torch.uint8)
codes |= torch.randint(0, 2, (n, k), dtype=torch.uint8) * 128
weight = codes.view(torch.float8_e4m3fn)
scales = torch.full((27, 48), 1 / 64, dtype=torch.float32)
prepared = prepare_block_fp8(weight, scales, torch.zeros(n))
layer = types.SimpleNamespace(prefix='model.layers.0.self_attn.qkv_proj',
                             _gk_block_fp8_selected=True,
                             _gk_block_fp8=prepared.to('hpu'))
torch.save(dict(weight=weight, scales=scales, gamma=gamma0,
                prepared_bytes=prepared.weight.view(torch.uint8),
                prepared_scales=prepared.scales), out / 'weights.pt')
sync()
records = []
cases = [(m, shape, 'random', False) for m, shape in
         [(1, '2d'), (2, '3d'), (8, '2d'), (513, '3d')]]
cases += [(1, '2d', name, False) for name in ('zero', 'signed_zero', 'product_rounding')]
cases += [(m, '2d', pattern, True) for m, pattern in
          [(1, 'cancellation'), (2, 'random'), (8, 'random'), (513, 'random')]]
with torch.inference_mode():
    for m, layout, pattern, has_residual in cases:
        name = f'm{m}-{layout}-{pattern}-res{int(has_residual)}'
        dest = out / name
        dest.mkdir()
        shape = (m, k) if layout == '2d' else (1, m, k)
        x0 = torch.randn(shape).bfloat16()
        r0 = torch.randn(shape).bfloat16() if has_residual else None
        if pattern in ('zero', 'signed_zero'):
            x0.zero_()
            if pattern == 'signed_zero':
                x0[..., ::2] = -0.0
        elif pattern == 'product_rounding':
            x0.fill_(1.0078125)
            x0[..., 1::2] *= -1
        elif pattern == 'cancellation':
            r0 = -x0
            r0[..., ::7] += torch.tensor(.0078125, dtype=torch.bfloat16)
        x = x0.to('hpu')
        r = r0.to('hpu') if r0 is not None else None
        sync()
        saved = dict(x=x0, residual=r0)
        torch.save(saved, dest / 'raw.pt')
        policies = ['bf16_fp32'] + (['decode_a8_bf16_fp32'] if m == 1 else [])

        def produce():
            # A real pointwise producer excludes a persistent-input-only gate;
            # signed-zero uses the original input to isolate the norm from
            # the upstream vendor multiply's own zero-sign semantics.
            produced = x if pattern == 'signed_zero' else x * 1.0
            result = resident(produced, r)
            return result[0] if has_residual else result

        def check_norm(host_x, tag):
            produced = x if pattern == 'signed_zero' else x * 1.0
            result = resident(produced, r)
            sync()
            if has_residual:
                actual, residual_actual = result[0].cpu(), result[1].cpu()
                residual_expected = (host_x + r0).bfloat16()
                assert bits_equal(residual_actual, residual_expected), (name, tag, 'residual')
            else:
                actual = result.cpu()
                residual_expected = host_x
                # Separately materialized vendor-f32 diagnostic. Its output
                # lifetime/tree is not the custom TPC's pre-round instruction.
                wide = rms_norm().apply(produced.float(), resident.weight.float(),
                                        resident.variance_epsilon)
                sync()
                wide_cpu = wide.cpu()
                saved[tag + '_norm'] = actual
                saved[tag + '_pre_round_fp32'] = wide_cpu
                torch.save(saved, dest / 'raw.pt')
                assert wide_cpu.dtype == torch.float32
                del wide
            z = residual_expected.float()
            # Independent ordinary FP32 arithmetic; reduction order is CPU's.
            cpu = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + float(torch.tensor(1e-6))) * gamma0.float()
            # Bound ordinary finite reduction differences separately from the
            # graph's mandatory dtype audit. This is not a GPU-tree emulator.
            u = 2.0 ** -24
            gamma = (k - 1) * u / (1 - (k - 1) * u)
            relative_budget = gamma + 16 * u
            lo = (cpu - cpu.abs() * relative_budget).bfloat16().float()
            hi = (cpu + cpu.abs() * relative_budget).bfloat16().float()
            assert bool(((actual.float() >= lo) & (actual.float() <= hi)).all()), (name, tag, 'BF16 rounding cell')
            if not has_residual:
                scale = cpu.abs().clamp_min(torch.finfo(torch.float32).tiny)
                bound = scale * relative_budget
                assert bool(torch.isfinite(wide_cpu).all())
                assert bool(((wide_cpu - cpu).abs() <= bound).all()), (name, tag, 'FP32 norm bound')
                # A separately materialized FP32 output can change compiler
                # reduction/lowering. Compare with FP32 bounds and record its
                # final-BF16 disagreement; do not assume identical trees.
                rounded = wide_cpu.bfloat16()
                saved[tag + '_separate_fp32_comparison'] = dict(
                    bf16_mismatches=int((actual.view(torch.int16) != rounded.view(torch.int16)).sum()),
                    max_bf16_ulp=float(((actual.float() - rounded.float()).abs() /
                        torch.pow(2., torch.floor(torch.log2(rounded.float().abs().clamp_min(2.**-126))) - 7)).max()),
                    max_preround_relative_error=float(((wide_cpu - cpu).abs() / scale).max()))
                if pattern == 'signed_zero' and tag == 'initial':
                    saved[tag + '_separate_fp32_comparison']['zero_sign_differences'] = int(
                        (torch.signbit(actual) != torch.signbit(host_x * gamma0)).sum())
                    assert bool((actual == 0).all()), 'nonzero RMS of zero input'
            else:
                # The retained qualified TPC path's final BF16 output is
                # checked against cells enclosing the FP32 reference interval.
                assert bool(((actual.float() >= lo) & (actual.float() <= hi)).all())
            assert bool(torch.isfinite(actual.float()).all())
            saved.update({tag + '_norm': actual, tag + '_cpu_fp32': cpu,
                          tag + '_residual_boundary': residual_expected})
            torch.save(saved, dest / 'raw.pt')
            del result
            return actual, relative(actual.float(), cpu.bfloat16().float())

        norm0, normerr = check_norm(x0, 'initial')
        for policy in policies:
            state.set_policy(policy)
            norm_input = norm0.to('hpu')
            sync()
            expected = method.apply(layer, norm_input)
            sync()
            expected0 = expected.cpu()
            del expected
            stream, graph = torch.hpu.Stream(), torch.hpu.HPUGraph()
            with torch.hpu.graph(graph, stream=stream):
                output = method.apply(layer, produce()) + 0.0
            sync()
            graph.replay(asynchronous=True)
            sync()
            actual = output.cpu()
            # +0 is only a downstream producer-lifetime consumer; it may
            # canonicalize output signed zero in both comparison arms.
            expected0 = (expected0 + 0.0).bfloat16()
            saved[policy] = dict(actual=actual, same_norm_input_control=expected0)
            torch.save(saved, dest / 'raw.pt')
            assert bits_equal(actual, expected0), (name, policy, 'initial QKV boundary')
            address = output.data_ptr()
            changed_x = -x0 if pattern not in ('zero', 'signed_zero') else torch.full_like(x0, .25)
            x.copy_(changed_x)
            sync()
            norm1, changed_normerr = check_norm(changed_x, 'changed')
            norm_input.copy_(norm1)
            sync()
            expected = method.apply(layer, norm_input)
            sync()
            expected1 = (expected.cpu() + 0.0).bfloat16()
            del expected
            graph.replay(asynchronous=True)
            sync()
            changed = output.cpu()
            saved[policy].update(changed=changed, changed_same_norm_input_control=expected1)
            torch.save(saved, dest / 'raw.pt')
            assert output.data_ptr() == address and bits_equal(changed, expected1), (name, policy, 'changed QKV boundary')
            times = []
            for _ in range(3):
                start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                begin = time.perf_counter()
                with torch.hpu.stream(stream):
                    start.record(stream)
                    for _ in range(20):
                        graph.replay(asynchronous=True)
                    end.record(stream)
                end.synchronize()
                sync()
                times.append(dict(event_us=start.elapsed_time(end) * 1000 / 20,
                                  wall_us=(time.perf_counter() - begin) * 1e6 / 20))
            timing_qualified = all(abs(t['event_us'] - t['wall_us']) / t['wall_us'] < .25 for t in times)
            saved[policy] = dict(actual=actual, same_norm_input_control=expected0,
                                 changed=changed, changed_same_norm_input_control=expected1)
            torch.save(saved, dest / 'raw.pt')
            record = dict(name=name, policy=policy, pure=not has_residual,
                          norm_relative_l2_to_cpu_fp32_bf16=normerr,
                          changed_norm_relative_l2=changed_normerr,
                          qkv_equal_bits=True, changed_qkv_equal_bits=True,
                          output_address_stable=True, timing=times,
                          timing_event_wall_consistent=timing_qualified,
                          separate_fp32_comparison={tag: saved.get(tag + '_separate_fp32_comparison')
                                                    for tag in ['initial', 'changed']})
            records.append(record)
            print(json.dumps(record), flush=True)
            (out / 'result.json').write_text(json.dumps(dict(status='INCOMPLETE_GRAPH_AUDIT', records=records), indent=2) + '\n')
            del graph, output, norm_input
            x.copy_(x0)
            sync()
        del x, r
        sync()

(out / 'result.json').write_text(json.dumps(dict(
    status='PASS_DEVICE_VALUES_PENDING_GRAPH_AUDIT', records=records,
    installation=installation,
    libraries={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
               for p in [a.norm_extension, a.block_extension]},
    scope='Norm FP32 dtype/rounding and real-wrapper-to-QKV ownership/replay; same actual norm-input QKV control; not a full-model or independent MME certificate'), indent=2) + '\n')
