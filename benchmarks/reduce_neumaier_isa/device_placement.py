"""Paired public-graph placement ablation; run only via launch_placement.py."""
import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--assets', type=Path, required=True)
a = p.parse_args()
assert os.environ.get('GAUDI_KERNELS_MODULE_ID') == '7'
assert os.environ.get('HABANA_VISIBLE_MODULES') == '7'
out = Path(os.environ['PROBE_OUT']) / 'placement'
out.mkdir(parents=True, exist_ok=False)
(out / 'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true', GRAPH_VISUALIZATION='1',
                  SRAM_SLICER_GRAPH_VISUALIZATION='1', LOG_LEVEL_GC='0',
                  GRAPH_VISUALIZATION_DIR=str(out / 'graphs'),
                  DUMP_POST_GRAPHS=str(out / 'post_graph.json'))
import numpy as np
import torch
import habana_frameworks.torch.core as hc

torch.set_num_threads(4)
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'benchmarks/block_fp8_framework'))
from oracle import errors

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((a.assets / 'manifest.json').read_text())
for name, digest in manifest['files_sha256'].items():
    assert sha(a.assets / name) == digest, name
for name in ('old-torch.so', 'hand-torch.so', 'block-torch.so'):
    torch.ops.load_library(str((a.assets / name).resolve()))
counter = ctypes.CDLL(None).gaudi_kernel_launch_count
counter.restype = ctypes.c_ulonglong
report = dict(status='INCOMPLETE', device_tested=True, private_launch=False,
              timing_requested=False, production_default_changed=False,
              physical_HBM_measured=False, asset_manifest_sha256=sha(a.assets / 'manifest.json'),
              captures=[], numerics=[], quant_bit_gates=[])


def save():
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


def sync():
    hc.mark_step()
    torch.hpu.synchronize()


def raw(t):
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def capture(fn):
    stream, graph = torch.hpu.Stream(), torch.hpu.HPUGraph()
    with torch.hpu.graph(graph, stream=stream):
        y = fn()
    sync()
    graph.replay(asynchronous=True)
    sync()
    return graph, stream, y


held = []
try:
    save()
    for name in ('qkv-measured-m1', 'single-group-m16'):
        fixture = torch.load(a.assets / (name + '.pt'), map_location='cpu', weights_only=False)
        m, n, k = fixture['shape']
        sample = fixture['samples'][0]
        pw = fixture['weight_bits'].view(torch.float8_e4m3fn).to('hpu')
        sw, bias = fixture['weight_scales'].to('hpu'), fixture['bias'].to('hpu')
        x = sample['x'].to('hpu')
        q_external = sample['quant_bits'].view(torch.float8_e4m3fn).to('hpu')
        sa_external = sample['activation_scales'].to('hpu')
        sync()
        graphs = {}
        for mode in ('quant_in_graph', 'external_quant'):
            for variant, reducer in (('old', torch.ops.gk_reduce_experiment.neumaier),
                                     ('hand', torch.ops.gk_reduce_isa.handschedule)):
                def apply():
                    q, sa = torch.ops.gaudi_block_fp8.quant(x) if mode == 'quant_in_graph' else (q_external, sa_external)
                    partial = torch.ops.gaudi_block_fp8.batch_mm(q, pw)
                    return reducer(partial, sa, sw, bias)
                graphs[mode, variant] = capture(apply)
                report['captures'].append(dict(case=name, shape=[m, n, k], mode=mode, variant=variant,
                                               partial_logical_bytes=m*n*((k+127)//128)*4))
                save()
        for index, sample in enumerate(fixture['samples']):
            x.copy_(sample['x'])
            q_external.copy_(sample['quant_bits'].view(torch.float8_e4m3fn))
            sa_external.copy_(sample['activation_scales'])
            sync()
            # A separate diagnostic graph verifies the saved q/sa bits. Its
            # outputs never expose the partial inside any tested capture.
            q, sa = torch.ops.gaudi_block_fp8.quant(x)
            sync()
            same_q = raw(q.cpu()) == raw(sample['quant_bits']) == raw(q_external.cpu())
            same_sa = raw(sa.cpu()) == raw(sample['activation_scales']) == raw(sa_external.cpu())
            report['quant_bit_gates'].append(dict(case=name, input_index=index,
                                                  activation_bytes_identical=same_q, scale_bytes_identical=same_sa))
            save()
            assert same_q and same_sa, (name, index, 'quantization bits')
            outputs = {}
            for (mode, variant), (graph, stream, y) in graphs.items():
                before = counter()
                graph.replay(asynchronous=True)
                sync()
                launches = counter() - before
                got = y.cpu()
                value = raw(got)
                outputs[mode, variant] = value
                filename = f'{name}-{mode}-{variant}-input{index}.bin'
                (out / filename).write_bytes(value)
                e = errors(got, sample['refs'])
                gpu_error = errors(sample['refs']['gpu_style_fp64'].bfloat16(), sample['refs'])['high_fp64']['relative_l2']
                valid = e['finite'] and e['adapted_fp64']['relative_l2'] < .006 and e['high_fp64']['relative_l2'] <= 1.15*gpu_error+.008 and launches == 1
                row = dict(case=name, mode=mode, variant=variant, input_index=index,
                           checked=got.numel(), errors=e, original_quality_gate_pass=valid,
                           public_launches=launches, one_recipe_per_replay=launches == 1,
                           output_file=filename, output_sha256=hashlib.sha256(value).hexdigest())
                report['numerics'].append(row)
                save()
                assert valid, row
            assert len(set(outputs.values())) == 1, (name, index, 'paired full-output bit mismatch')
        held.append((graphs, (x, q_external, sa_external, pw, sw, bias)))
        assert (out / f'{name}-quant_in_graph-old-input0.bin').read_bytes() != (out / f'{name}-quant_in_graph-old-input1.bin').read_bytes()
    report.update(status='PASS_NUMERICS_PLACEMENT_UNAUDITED', numerics_pass=True,
                  placement_audited=False, all_requirements_pass=False,
                  checked_outputs=sum(x['checked'] for x in report['numerics']))
except BaseException as error:
    report.update(status='FAIL', numerics_pass=False, all_requirements_pass=False, error=repr(error))
    raise
finally:
    save()
print(json.dumps({k: report[k] for k in ('status', 'checked_outputs')}), flush=True)
