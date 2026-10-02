"""Four dependent public Linear calls, one replay: isolate the submission floor.

This is a synthetic operator chain, not a transformer or model TPS benchmark.
Distinct persistent weight owners represent four layers; every intermediate is
also retained for numerical inspection. No CPU code runs between its layers.
"""
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import time

out = Path(os.environ['PROBE_OUT'])
(out / 'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true', GRAPH_VISUALIZATION='1',
                  GRAPH_VISUALIZATION_DIR=str(out / 'graphs'),
                  DUMP_POST_GRAPHS=str(out / 'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from validation import timings_agree

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'python'))
from gaudi_kernels.torch_linear import load_extension, prepare_separable_fp8, linear_fp8
from gaudi_kernels.fp8_quantizer import prepare_fp8_quantizer

load_extension(root / os.environ['GK_TORCH_BUILD'] / 'gaudi_kernels_torch.so')
counter = ctypes.CDLL(None).gaudi_kernel_launch_count
counter.restype = ctypes.c_ulonglong
torch.set_num_threads(4)
torch.manual_seed(270931)
m, n, k, depth = 512, 2048, 2048, 4
plans = {v: prepare_fp8_quantizer(v).to('hpu') for v in ['lut1', 'lut4']}
cpu_weights = []
saved = {'shape': [m, n, k], 'depth': depth, 'layers': []}
for _ in range(depth):
    raw = (torch.randn(n, k) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    scale = torch.rand(n) * .0002 + .0002
    bias = torch.randn(n) * .01
    prepared = prepare_separable_fp8(raw, scale, bias)
    cpu_weights.append(prepared)
    saved['layers'].append({'checkpoint': raw, 'scale': scale, 'bias': bias,
                           'prepared_bytes': prepared.weight.view(torch.uint8),
                           'prepared_scale': prepared.scale})
x0 = torch.randn(m, k).bfloat16()
saved['input'] = x0
torch.save(saved, out / 'inputs-outputs-oracles.pt')
weights = [w.to('hpu') for w in cpu_weights]
x = x0.to('hpu')


def sync():
    hc.mark_step()
    torch.hpu.synchronize()


def check_stages(outputs, source):
    """Check each MAC against FP64 using that stage's actual BF16 input."""
    errors, references = [], []
    for value, weight in zip(outputs, cpu_weights):
        scale = source.float().abs().amax(-1, keepdim=True).clamp_min(1e-10) * (1 / 448)
        q = ((source.float() / scale).clamp(-448, 448)
             .to(torch.float8_e4m3fn).float() * .5).to(torch.float8_e4m3fn)
        reference = ((q.double() * (scale.double() * 2)) @ weight.weight.double().T
                     * weight.scale.double() + weight.bias.double()).bfloat16()
        error = float((value.double() - reference.double()).norm()
                      / reference.double().norm().clamp_min(1e-30))
        assert bool(torch.isfinite(value).all()) and error < .001
        errors.append(error)
        references.append(reference)
        source = value
    return errors, references


sync()
records, baseline = [], None
for variant in ['existing', 'lut1', 'lut4']:
    x.copy_(x0)
    sync()
    stream, graph = torch.hpu.Stream(), torch.hpu.HPUGraph()
    with torch.hpu.graph(graph, stream=stream):
        y, ys = x, []
        for weight in weights:
            y = linear_fp8(y, weight, activation='per_token_fp8', optimized=True,
                           quantization=plans.get(variant))
            ys.append(y)
    sync()
    graph.replay(asynchronous=True)
    sync()
    actual = [y.cpu() for y in ys]
    errors, references = check_stages(actual, x0)
    if baseline is None:
        baseline = actual
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16))
               for a, b in zip(actual, baseline))
    for _ in range(5):
        graph.replay(asynchronous=True)
    sync()
    before = counter()
    for _ in range(10):
        graph.replay(asynchronous=True)
    sync()
    launches = counter() - before
    assert launches == 10
    event, wall = [], []
    for _ in range(5):
        begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record(stream)
        for _ in range(50):
            graph.replay(asynchronous=True)
        sync()
        end.record(stream)
        end.synchronize()
        sync()
        event.append(begin.elapsed_time(end) * 1000 / 50)
        wall.append((time.perf_counter() - start) * 1e6 / 50)
    assert timings_agree(event, wall)
    pointers = [y.data_ptr() for y in ys]
    x.copy_(x0.neg())
    sync()
    graph.replay(asynchronous=True)
    sync()
    changed = [y.cpu() for y in ys]
    changed_errors, changed_references = check_stages(changed, x0.neg())
    assert pointers == [y.data_ptr() for y in ys]
    assert all(not torch.equal(a, b) for a, b in zip(actual, changed))
    saved[variant] = {'actual': actual, 'references': references,
                      'changed': changed, 'changed_references': changed_references}
    torch.save(saved, out / 'inputs-outputs-oracles.pt')
    row = {'M': m, 'N': n, 'K': k, 'depth': depth, 'variant': variant,
           'checked_outputs': sum(a.numel() for a in actual), 'bit_mismatches': 0,
           'stage_relative_l2': errors, 'changed_stage_relative_l2': changed_errors,
           'stable_pointers': True, 'launches_per_10_replays': launches,
           'event_us': event, 'wall_us': wall,
           'median_event_us': statistics.median(event), 'median_wall_us': statistics.median(wall)}
    records.append(row)
    print(json.dumps(row), flush=True)
    (out / 'result.json').write_text(json.dumps({'status': 'INCOMPLETE', 'records': records}, indent=2) + '\n')
(out / 'result.json').write_text(json.dumps({'status': 'PASS', 'records': records,
    'scope': 'four dependent Linear operators in one graph; not model quality/TPS'}, indent=2) + '\n')
