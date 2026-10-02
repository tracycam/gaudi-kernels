"""Current production compact versus device-routed expert pipeline, same inputs.

The frozen exact-dot fixture isolates graph/layout bugs from legal FP32 error.
Real checkpoint and model quality remain separate gates. No CPU route readback
is used to construct or replay any graph; counts below are diagnostics only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--fixture', type=Path, required=True)
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--environment', type=Path, required=True)
p.add_argument('--views-library', type=Path, required=True)
p.add_argument('--stream-tpc-library', type=Path)
p.add_argument('--stream-decoder', choices=('qualified_u4','k8'), default='qualified_u4')
p.add_argument('--tokens', type=int, nargs='+', default=[1, 2, 3, 5, 8])
p.add_argument('--modes', nargs='+', choices=('production', 'staged', 'expert_stream'), default=['production', 'staged', 'expert_stream'])
p.add_argument('--replays', type=int, default=20)
a = p.parse_args()
assert all(1 <= t <= 8 for t in a.tokens) and 1 <= a.replays <= 100
out = Path(os.environ['PROBE_OUT'])
env = json.loads(a.environment.read_text())
for k, v in env.items():
    if k.startswith('GK_MXFP4_') or k.startswith('GK_MOE_ROUTE_'):
        os.environ[k] = v
os.environ.update(PT_ENABLE_INT64_SUPPORT='0', UNIFIED_PRECISION_ROUTER='1',
                  PT_HPU_LAZY_MODE='1', ENABLE_EXPERIMENTAL_FLAGS='true',
                  GRAPH_VISUALIZATION='1', GRAPH_VISUALIZATION_DIR=str(out/'graphs'),
                  DUMP_POST_GRAPHS=str(out/'post_graph.json'))
runtime = a.runtime.resolve()
tpc = [runtime/'executor/tpc/libnative_tpc.so', runtime/'tpc/libbatch_tpc.so',
       runtime/'precision-tpc/libprecision_tpc.so']
for family in ('FOLDED', 'DOWN', 'SCALE_TAIL'):
    assert env['GK_MXFP4_'+family+'_ENABLED'] == '1'
    tpc.append(Path(env['GK_MXFP4_'+family+'_TPC_LIBRARY']))
tpc.append(Path(env['GK_MOE_ROUTE_TPC_LIBRARY']))
os.environ['GC_KERNEL_PATH'] = ':'.join(map(str, [Path('/usr/lib/habanalabs/libtpc_kernels.so'), *tpc]))
if a.stream_decoder == 'k8':
    assert a.stream_tpc_library is not None
    if str(a.stream_tpc_library.resolve()) not in os.environ['GC_KERNEL_PATH'].split(':'):
        os.environ['GC_KERNEL_PATH'] += ':'+str(a.stream_tpc_library.resolve())
import numpy as np
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0, str(runtime/'executor'))
import native_ops
import batch_ops
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.mxfp4_route_tiles import moe_route_tiles
for key in ('CORE', 'METADATA', 'TILES'):
    torch.ops.load_library(env['GK_MOE_ROUTE_'+key+'_TORCH_LIBRARY'])
torch.ops.load_library(str(a.views_library.resolve()))
batch_ops.set_gp_policy('vector_fetch_scale_tail')
batch_ops.set_down_policy('vector_fetch')
torch.set_num_threads(2)
result = dict(stream_decoder=a.stream_decoder, status='CHECKING', checks=[], timing=[], geometry=[],
              libraries={str(v): hashlib.sha256(v.read_bytes()).hexdigest() for v in tpc},
              scope='Public HPU graphs, same historical owners; not native model TPS or physical bandwidth')
def save():
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
def sync():
    hc.mark_step()
    torch.hpu.synchronize()
def bits(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))

with torch.inference_mode():
    weights = [torch.from_numpy(np.load(a.fixture/(name+'.npy'))).to('hpu')
               for name in ('gp', 'gps', 'down', 'downs')]
    table, directions = native_ops.constants('hpu')
    sync()
    for t in a.tokens:
        inputs = {n: torch.from_numpy(np.load(a.fixture/'uniform'/(n+'.npy'))[:t].copy()).to(
            dtype=torch.bfloat16 if n == 'x' else torch.int32 if n == 'ids' else torch.float32,
            device='hpu') for n in ('x', 'ids', 'routing')}
        sync()
        graphs, outputs, streams = {}, {}, {}
        for mode in a.modes:
            graph, stream = torch.hpu.HPUGraph(), torch.hpu.Stream()
            with torch.hpu.graph(graph, stream=stream):
                x = inputs['x'] + 0
                ids, routing = inputs['ids'], inputs['routing']
                if mode == 'production':
                    y = batch_ops.moe(x, ids, routing, *weights, table, directions, mode='compact')
                else:
                    y, meta = moe_route_tiles(x.clone(), ids.clone(), routing.clone(), *weights, table, layout=1,
                        rows=t, n_tile=2048, metadata_version=3, schedule=mode, stream_decoder=a.stream_decoder)
                consumer = y + .03125
            sync()
            graphs[mode], streams[mode], outputs[mode] = graph, stream, (y, consumer)
        for state in ('uniform', 'hot', 'skew', 'zero', 'restored'):
            path = a.fixture/('uniform' if state == 'restored' else state)
            for name in inputs:
                inputs[name].copy_(torch.from_numpy(np.load(path/(name+'.npy'))[:t].copy()))
            sync()
            ids_cpu = np.load(path/'ids.npy')[:t]
            result['geometry'].append(dict(T=t, state=state, active_experts=int(np.unique(ids_cpu).size),
                counts=np.unique(ids_cpu, return_counts=True)[1].tolist(), capacity=t*8))
            expected = np.load(path/'expected.npy')[:t]
            absolute = np.load(path/'absolute.npy')[:t]
            u = 2**-24
            bound = ((256+8)*u/(1-(256+8)*u))*absolute + 4*np.abs(np.spacing(expected)).astype(np.float64)
            actual = {}
            for mode in graphs:
                graphs[mode].replay(asynchronous=True)
                sync()
                y, c = (v.cpu() for v in outputs[mode])
                actual[mode] = y
                delta = np.abs(y.numpy().astype(np.float64)-expected)
                bad = int(np.count_nonzero(~np.isfinite(y.numpy()) | (delta > bound)))
                check = dict(T=t, state=state, mode=mode, bad=bad,
                             consumer_bits_equal=bits(c, y+.03125), max_abs=float(delta.max()))
                torch.save(dict(y=y, consumer=c), out/f't{t}-{state}-{mode}.pt')
                result['checks'].append(check)
                save()
                assert bad == 0 and check['consumer_bits_equal'], check
            same = bits(actual['staged'], actual['expert_stream']) if 'staged' in actual else None
            result['checks'].append(dict(T=t, state=state, stream_vs_stage_bitwise=same,
                production_vs_stream_bitwise=bits(actual['production'], actual['expert_stream'])))
            # The down-N MME tiling differs; legal FP32 partial-sum changes
            # are covered by the independently checked bound above.
            if state not in ('uniform', 'hot'):
                continue
            for mode in graphs:
                for _ in range(3):
                    graphs[mode].replay(asynchronous=True)
                sync()
            for arm, mode in enumerate(a.modes+list(reversed(a.modes))):
                for trial in range(3):
                    stream = streams[mode]
                    start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    with torch.hpu.stream(stream):
                        wall = time.perf_counter()
                        start.record(stream)
                        for _ in range(a.replays):
                            graphs[mode].replay(asynchronous=True)
                        end.record(stream)
                    end.synchronize()
                    result['timing'].append(dict(T=t, state=state, mode=mode, arm=arm, trial=trial,
                        replays=a.replays, event_us=start.elapsed_time(end)*1000/a.replays,
                        wall_us=(time.perf_counter()-wall)*1e6/a.replays))
                    save()
        del graphs, outputs, streams, inputs
        sync()
    result.update(status='PASS_SAME_INPUT_PRODUCTION_STREAM', production_gp=batch_ops.gp_snapshot(),
                  production_down=batch_ops.down_snapshot())
    save()
