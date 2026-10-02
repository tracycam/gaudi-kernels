"""Complete production MoE ABBA with only the GP instruction schedule changed."""
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
p.add_argument('--build', type=Path, required=True)
p.add_argument('--tokens', type=int, nargs='+', default=[1, 2, 3, 4, 8])
p.add_argument('--replays', type=int, default=40)
a = p.parse_args()
assert all(1 <= t <= 8 for t in a.tokens) and 1 <= a.replays <= 100
out = Path(os.environ['PROBE_OUT'])
env = json.loads(a.environment.read_text())
for k, v in env.items():
    if k.startswith('GK_MXFP4_'):
        os.environ[k] = v
os.environ.update(PT_ENABLE_INT64_SUPPORT='0', UNIFIED_PRECISION_ROUTER='1',
                  UNIFIED_PRECISION_GATE='1', PT_HPU_LAZY_MODE='1', ENABLE_EXPERIMENTAL_FLAGS='true',
                  GRAPH_VISUALIZATION='1', GRAPH_VISUALIZATION_DIR=str(out/'graphs'),
                  DUMP_POST_GRAPHS=str(out/'post_graph.json'))
runtime = a.runtime.resolve()
tpc = [runtime/'executor/tpc/libnative_tpc.so', runtime/'tpc/libbatch_tpc.so', runtime/'precision-tpc/libprecision_tpc.so']
for family in ('FOLDED', 'DOWN', 'SCALE_TAIL'):
    assert env['GK_MXFP4_'+family+'_ENABLED'] == '1'
    tpc.append(Path(env['GK_MXFP4_'+family+'_TPC_LIBRARY']))
tpc.append(a.build.resolve()/'libgaudi_gp_prefetch_tpc.so')
os.environ['GC_KERNEL_PATH'] = ':'.join(map(str, [Path('/usr/lib/habanalabs/libtpc_kernels.so'), *tpc]))
import numpy as np
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0, str(runtime/'executor'))
import native_ops
import batch_ops
from precision_ops import combine
torch.ops.load_library(str(a.build.resolve()/'gaudi_gp_prefetch.so'))
batch_ops.set_gp_policy('vector_fetch_scale_tail')
batch_ops.set_down_policy('vector_fetch')
torch.set_num_threads(2)
result = dict(status='CHECKING', checks=[], timing=[], geometry=[], graphs=[],
              libraries={str(v): hashlib.sha256(v.read_bytes()).hexdigest() for v in tpc},
              scope='Complete TP-local production MoE with same materialization/gate/down/combine; no model TPS claim')
def save():
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
def sync():
    hc.mark_step(); torch.hpu.synchronize()
def bits(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
def candidate(x, ids, routing, weights, table, directions):
    # Match the existing production boundaries exactly; swap only the GP GUID.
    x=x.clone(); ids=ids.to(torch.int32).clone(); routing=routing.to(torch.float32).clone()
    gp, gs, dp, ds = weights
    partial = torch.ops.gaudi_gp_prefetch.gp(gp, gs, x, table, ids)
    gate = torch.ops.unified_batch.gate(partial, ids)
    down = torch.ops.gaudi_down_activation.broadcast(dp, ds, gate, table, ids)
    return combine(down, routing, directions, 6144, ids.shape[1])

with torch.inference_mode():
    weights = [torch.from_numpy(np.load(a.fixture/(name+'.npy'))).to('hpu') for name in ('gp','gps','down','downs')]
    table, directions = native_ops.constants('hpu'); sync()
    for t in a.tokens:
        inputs = {n: torch.from_numpy(np.load(a.fixture/'uniform'/(n+'.npy'))[:t].copy()).to(
            dtype=torch.bfloat16 if n=='x' else torch.int32 if n=='ids' else torch.float32, device='hpu') for n in ('x','ids','routing')}
        sync(); graphs={}; outputs={}; streams={}
        for mode in ('production', 'prefetch'):
            graph, stream = torch.hpu.HPUGraph(), torch.hpu.Stream()
            with torch.hpu.graph(graph, stream=stream):
                x = inputs['x'] + 0
                if mode == 'production':
                    y = batch_ops.moe(x, inputs['ids'], inputs['routing'], *weights, table, directions, mode='compact')
                else:
                    y = candidate(x, inputs['ids'], inputs['routing'], weights, table, directions)
                consumer = y + .03125
            sync(); graphs[mode]=graph; streams[mode]=stream; outputs[mode]=(y,consumer)
        for state in ('uniform','hot','mixed','zero','restored'):
            path = a.fixture/('uniform' if state in ('restored','mixed') else state)
            raw={name:np.load(path/(name+'.npy'))[:t].copy()for name in inputs}
            if state=='mixed':
                # Slicing the old T513 skew fixture produces all-hot routes
                # at T<=8. Construct genuine mixed counts at this T instead.
                base=np.array([[0,1,2,3,4,5,6,7],[0,1,2,3,8,9,10,11],
                               [0,1,2,3,12,13,14,15],[4,5,6,7,16,17,18,19]],np.int32)
                raw['ids']=np.stack([base[i%4]+20*(i//4)for i in range(t)])
            for name in inputs:
                inputs[name].copy_(torch.from_numpy(raw[name]))
            sync()
            ids = raw['ids']
            result['geometry'].append(dict(T=t,state=state,active_experts=int(np.unique(ids).size),counts=np.unique(ids,return_counts=True)[1].tolist()))
            expected = np.load(path/'expected.npy')[:t]
            absolute = np.load(path/'absolute.npy')[:t]
            if state=='mixed':
                dot=np.load(a.fixture/'expert_down_exact.npy',mmap_mode='r')
                absdot=np.load(a.fixture/'expert_down_absolute.npy',mmap_mode='r')
                expected=np.zeros((t,6144),np.float32);absolute=np.zeros((t,6144),np.float64)
                for slot in range(8):
                    term=dot[ids[:,slot],np.arange(t)%2]
                    route=raw['routing'][:,slot,None]
                    expected=(term.astype(np.float32).astype(np.float64)*route+expected.astype(np.float64)).astype(np.float32)
                    absolute+=absdot[ids[:,slot],np.arange(t)%2]*np.abs(route)
                torch.save({k:torch.from_numpy(v)for k,v in raw.items()},out/f't{t}-mixed-inputs.pt')
            u=2**-24; bound=((256+8)*u/(1-(256+8)*u))*absolute+4*np.abs(np.spacing(expected)).astype(np.float64)
            actual={}
            for mode in graphs:
                graphs[mode].replay(asynchronous=True);sync()
                y,c=(v.cpu() for v in outputs[mode]);actual[mode]=(y,c)
                delta=np.abs(y.numpy().astype(np.float64)-expected)
                bad=int(np.count_nonzero(~np.isfinite(y.numpy())|(delta>bound)))
                record=dict(T=t,state=state,mode=mode,bad=bad,consumer_bits_equal=bits(c,y+.03125),max_abs=float(delta.max()))
                result['checks'].append(record);torch.save(dict(y=y,consumer=c),out/f't{t}-{state}-{mode}.pt');save()
                assert bad==0 and record['consumer_bits_equal'],record
            equal=all(bits(x,y) for x,y in zip(actual['production'],actual['prefetch']))
            result['checks'].append(dict(T=t,state=state,production_prefetch_bitwise=equal));save();assert equal
            for mode in graphs:
                graphs[mode].replay(asynchronous=True);sync()
                assert all(bits(v.cpu(),ref) for v,ref in zip(outputs[mode],actual[mode]))
            if state not in ('uniform','hot','mixed'):continue
            for graph in graphs.values():
                for _ in range(4):graph.replay(asynchronous=True)
            sync()
            for trial in range(3):
                for arm,mode in enumerate(('production','prefetch','prefetch','production')):
                    stream=streams[mode];start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
                    with torch.hpu.stream(stream):
                        wall=time.perf_counter();start.record(stream)
                        for _ in range(a.replays):graphs[mode].replay(asynchronous=True)
                        torch.hpu.synchronize();end.record(stream);end.synchronize()
                    record=dict(T=t,state=state,mode=mode,trial=trial,arm=arm,replays=a.replays,
                                event_us=start.elapsed_time(end)*1000/a.replays,wall_us=(time.perf_counter()-wall)*1e6/a.replays)
                    result['timing'].append(record);save();assert .5<record['event_us']/record['wall_us']<1.2
        del graphs,outputs,streams,inputs;sync()
    result.update(status='PASS_FULL_MOE_BITS_ABBA',production_gp=batch_ops.gp_snapshot(),production_down=batch_ops.down_snapshot());save()
