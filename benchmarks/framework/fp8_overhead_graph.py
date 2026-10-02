"""Same prepared weights: baseline/optimized complete HPU Graph replay gates."""
import ctypes
import hashlib
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
library = root / os.environ['GK_TORCH_BUILD'] / 'gaudi_kernels_torch.so'
load_extension(library)
torch.set_num_threads(4)
torch.manual_seed(270927)
counter = ctypes.CDLL(None).gaudi_kernel_launch_count
counter.restype = ctypes.c_ulonglong


def sync():
    hc.mark_step()
    torch.hpu.synchronize()


records = []
saved = {}
for m, n, k in [(3, 129, 257), (512, 1024, 2048), (513, 1024, 2048)]:
    original = (torch.randn(n, k) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    scales = torch.rand(n) * .02 + .01
    bias = torch.randn(n) * .1
    packed = prepare_separable_fp8(original, scales, bias)
    device = packed.to('hpu')
    x0 = torch.randn(m, k).bfloat16()
    x = x0.to('hpu')
    sync()
    key = f'{m}-{n}-{k}'
    saved[key] = {'checkpoint': original, 'weight_scales': scales, 'bias': bias,
                  'native': packed.weight, 'native_scales': packed.scale, 'input': x0}
    for policy in ['per_token_fp8', 'bf16']:
        activ = x0.double()
        if policy == 'per_token_fp8':
            sa = x0.float().abs().amax(-1, keepdim=True).clamp_min(1e-10) * (1/448)
            qa = (x0.float()/sa).clamp(-448, 448).to(torch.float8_e4m3fn).float()
            activ = (qa*.5).to(torch.float8_e4m3fn).double() * (sa.double()*2)
        oracle = (activ @ packed.weight.double().T * packed.scale.double() + bias.double()).bfloat16()
        outputs = {}
        for optimized in [False, True]:
            x.copy_(x0)
            sync()
            stream = torch.hpu.Stream()
            graph = torch.hpu.HPUGraph()
            with torch.hpu.graph(graph, stream=stream):
                y = linear_fp8(x, device, activation=policy, optimized=optimized)
            sync()
            graph.replay(asynchronous=True)
            sync()
            actual = y.cpu()
            error = float((actual.double()-oracle.double()).norm()/oracle.double().norm())
            assert bool(torch.isfinite(actual).all()) and error < .001
            outputs[optimized] = actual
            saved[key][policy + str(optimized)] = actual
            saved[key][policy + '_oracle'] = oracle
            for _ in range(5): graph.replay(asynchronous=True)
            sync()
            before = counter()
            for _ in range(10): graph.replay(asynchronous=True)
            sync()
            launches = counter() - before
            assert launches == 10
            event, wall = [], []
            for _ in range(5):
                begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                start = time.perf_counter()
                begin.record(stream)
                for _ in range(80): graph.replay(asynchronous=True)
                sync()  # Drain the replay worker before ending the event interval.
                end.record(stream)
                end.synchronize()
                sync()
                event.append(begin.elapsed_time(end)*1000/80)
                wall.append((time.perf_counter()-start)*1e6/80)
            assert timings_agree(event, wall)
            pointer = y.data_ptr()
            x.copy_(x0.neg())
            sync()
            graph.replay(asynchronous=True)
            sync()
            changed = y.cpu()
            changed_ref = (-activ @ packed.weight.double().T * packed.scale.double() + bias.double()).bfloat16()
            changed_error = float((changed.double()-changed_ref.double()).norm()/changed_ref.double().norm())
            assert changed_error < .001 and y.data_ptr() == pointer and not torch.equal(changed, actual)
            saved[key][policy + str(optimized) + '_changed'] = changed
            record = {'M': m, 'N': n, 'K': k, 'activation': policy, 'optimized': optimized,
                      'full_outputs': m*n, 'relative_l2_to_rounded_contract': error,
                      'changed_input_relative_l2': changed_error, 'stable_output_address': True,
                      'synapse_launches_for_10_replays': launches, 'event_us': event, 'wall_us': wall,
                      'median_event_us': statistics.median(event), 'median_wall_us': statistics.median(wall)}
            records.append(record)
            print(json.dumps(record), flush=True)
        assert torch.equal(outputs[False], outputs[True]), 'optimized graph changed output bits'
torch.save(saved, out / 'inputs-outputs-oracles.pt')
(out / 'result.json').write_text(json.dumps({
    'status': 'PASS', 'records': records, 'baseline_optimized_bitwise_equal': True,
    'binding_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
    'scope': 'full HPU Graph operator replay, includes CPU scheduling; not whole-model TPS',
    'placement': 'requires postgraph audit for W8A16; numerical pass alone is insufficient'
}, indent=2) + '\n')
