"""Bounded view capture gate; run ONLY through run_device_probe.

Unsafe failures use separate processes. The safe suite acquires one device once
and stops at the first failure; no timing/production qualification is implied.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

CASES = ['noop', 'flatten3d', 'slice', 'transpose', 'recent_base', 'list']
DTYPES = ['bfloat16', 'float32', 'int32', 'int64']
p = argparse.ArgumentParser()
p.add_argument('--library', type=Path, required=True)
p.add_argument('--library-sha256', required=True)
p.add_argument('--case', choices=CASES + ['all'], required=True)
p.add_argument('--dtype', choices=DTYPES + ['all'], default='bfloat16')
p.add_argument('--safe', choices=['0', '1'], required=True)
a = p.parse_args()
if a.safe == '0' and (a.case == 'all' or a.dtype == 'all'):
    raise ValueError('unsafe baseline must be a single isolated fixture')
out = Path(os.environ['PROBE_OUT'])
if not os.environ.get('GAUDI_KERNELS_MODULE_ID') or os.environ.get('HABANA_VISIBLE_MODULES') != os.environ['GAUDI_KERNELS_MODULE_ID']:
    raise RuntimeError('bounded single-module runner required')
assert os.environ.get('PT_HPU_LAZY_MODE') == '1'
library = a.library.resolve(strict=True)
assert hashlib.sha256(library.read_bytes()).hexdigest() == a.library_sha256
(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true', DUMP_POST_GRAPHS=str(out/'post_graph.json'),
                  GRAPH_VISUALIZATION='1', GRAPH_VISUALIZATION_DIR=str(out/'graphs'),
                  PT_ENABLE_INT64_SUPPORT='0')
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
torch.set_num_threads(2)
torch.ops.load_library(str(library))
result = {'state': 'STARTED', 'safe': a.safe == '1', 'binary_sha256': a.library_sha256,
          'runtime_pin': torch.ops.gaudi_view_safe.runtime_pin(), 'records': [],
          'scope': 'private-ABI view fixture; not MoE MAC/production/performance qualification'}
def save():
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
def sync():
    hc.mark_step()
    torch.hpu.synchronize()
def views(x, case):
    produced = x + 1
    if case == 'noop':
        return [produced.view(*produced.shape)]
    if case == 'flatten3d':
        return [produced.reshape(1, 2, 8).reshape(2, 8)]
    if case == 'slice':
        return [produced[:, 1::2]]
    if case == 'transpose':
        return [produced.t()]
    if case == 'recent_base':
        view = produced.view(*produced.shape)
        produced.add_(3)
        return [view]
    return [produced[:, ::2], (produced + 2).t()]
def run_one(case, dtype_name):
    dest = out/(case+'-'+dtype_name)
    dest.mkdir()
    record = {'case': case, 'dtype': dtype_name, 'state': 'STARTED'}
    result['records'].append(record)
    dtype = getattr(torch, dtype_name)
    x0 = torch.arange(-8, 8).to(dtype).reshape(2, 8)
    changed0 = x0 + 7
    expected = [v.reshape(1, -1) + 3 for v in views(x0, case)]
    changed_expected = [v.reshape(1, -1) + 3 for v in views(changed0, case)]
    fixture = {'x': x0, 'changed_x': changed0, 'expected': expected, 'changed_expected': changed_expected}
    torch.save(fixture, dest/'fixture.pt')
    save()
    with torch.inference_mode():
        x = x0.to('hpu')
        sync()
        def chain():
            items = views(x, case)
            if case == 'list' and a.safe == '1':
                items = torch.ops.gaudi_view_safe.normalize_list(items)
            return [torch.ops.gaudi_view_safe.reshape(
                v, [1, v.numel()], a.safe == '1' and case != 'list') + 3 for v in items]
        stream, graph = torch.hpu.Stream(), torch.hpu.HPUGraph()
        with torch.hpu.graph(graph, stream=stream):
            outputs = chain()
        sync()
        owners = {'input': x.data_ptr(), 'outputs': [y.data_ptr() for y in outputs]}
        graph.replay(asynchronous=True)
        sync()
        actual = [y.cpu() for y in outputs]
        fixture['actual'] = actual
        torch.save(fixture, dest/'fixture.pt')
        assert all(torch.equal(y, ref) and y.dtype == ref.dtype for y, ref in zip(actual, expected))
        x.copy_(changed0)
        sync()
        # Same live graph/input/output owners. No allocation or readback in the
        # replay loop. data_ptr on Lazy tensors is logical, not physical proof.
        for _ in range(10):
            graph.replay(asynchronous=True)
        sync()
        assert owners == {'input': x.data_ptr(), 'outputs': [y.data_ptr() for y in outputs]}
        changed = [y.cpu() for y in outputs]
        fixture['changed_actual'] = changed
        torch.save(fixture, dest/'fixture.pt')
        assert all(torch.equal(y, ref) and y.dtype == ref.dtype for y, ref in zip(changed, changed_expected))
        assert any(not torch.equal(y, old) for y, old in zip(changed, actual))
        record.update(state='PASS_CAPTURE_AND_CHANGED_INPUT', checked_values=sum(y.numel() for y in actual)*2,
                      output_dtypes=[str(y.dtype) for y in actual], intermediate_views_retained=False,
                      changed_input_replays=10, logical_owner_handles=owners,
                      physical_address_stability='not directly inspected; graph-owned recipe replay only')
        del graph, outputs, x
        sync()
    (dest/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    save()
    print(json.dumps(record), flush=True)
try:
    for case in CASES if a.case == 'all' else [a.case]:
        for dtype in DTYPES if a.dtype == 'all' else [a.dtype]:
            run_one(case, dtype)
    result['state'] = 'PASS_CAPTURE_AND_CHANGED_INPUT'
except Exception as error:
    result.update(state='FAIL', error=repr(error))
    if result['records']:
        result['records'][-1].update(state='FAIL', error=repr(error))
    raise
finally:
    save()
print(json.dumps(result), flush=True)
