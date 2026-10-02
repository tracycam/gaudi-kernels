"""Bounded whole-MoE exact-GP-lattice gate, under the exclusive module runner.

FP64 fixture values are used to derive a sum-of-absolute-products FP32 bound,
not to require arbitrary once-rounded equality. This is not a model quality
gate or a performance promotion. All original-weight owners remain uint8.
"""
import argparse, hashlib, json, os, statistics, sys, time
from pathlib import Path

p=argparse.ArgumentParser()
for name in ('fixture','core-library','metadata-library','tiles-library'):
    p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--arm',choices=('u1','u2','u4'),required=True)
p.add_argument('--metadata-version',type=int,choices=(2,3),default=3)
p.add_argument('--rows',type=int,default=1);p.add_argument('--capacity',type=int)
p.add_argument('--n-tile',type=int,default=512);p.add_argument('--replays',type=int,default=4)
a=p.parse_args();assert 1<=a.replays<=16
module=os.environ.get('GAUDI_KERNELS_MODULE_ID')
assert module is not None and os.environ.get('HABANA_VISIBLE_MODULES')==module
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(PT_HPU_LAZY_MODE='1',ENABLE_EXPERIMENTAL_FLAGS='true',
    GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),
    DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import numpy as np
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels.mxfp4_route_tiles import moe_route_tiles
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/f'moe_route_metadata_v{a.metadata_version}'))
from reference import reference
torch.set_num_threads(2)
for path in (a.core_library,a.metadata_library,a.tiles_library):torch.ops.load_library(str(path.resolve()))
fixture=json.loads((a.fixture/'fixture.json').read_text())
assert fixture['GP_full_FP64_exactly_representable_in_FP32']
diverse=fixture['case']=='diverse_permuted'
assert diverse or (fixture['case']=='mixed' and fixture['E']==fixture['R']==2)
cpu={name:torch.from_numpy(np.load(a.fixture/(name+'.npy'))) for name in ('x','ids','routing','gp','gps','down','downs','lut')}
cpu['x']=cpu['x'].bfloat16();cpu['lut']=cpu['lut'].bfloat16()
inputs={n:v.to('hpu')for n,v in cpu.items()}
def sync():hc.mark_step();torch.hpu.synchronize()
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
names=('counts','row_status','prefix','tile_expert','tile_base','valid_rows','status','row_map','inverse','chunk_counts' if a.metadata_version==2 else 'chunk_offsets')
result=dict(status='CHECKING',arm=a.arm,metadata_version=a.metadata_version,module=module,fixture=fixture,checks=[],timing=[],
    libraries={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()for p in (a.core_library,a.metadata_library,a.tiles_library)},
    scope='Whole TP-local graph; exact GP lattice, literal gate, bounded legal FP32 down/reducer arithmetic. No model/TPS or SRAM placement qualification.')
with torch.inference_mode():
    sync();stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
    print('capture_begin',flush=True)
    with torch.hpu.graph(graph,stream=stream):
        produced=inputs['x']+0
        ids=inputs['ids']+0
        y,meta=moe_route_tiles(produced,ids,inputs['routing'],inputs['gp'],inputs['gps'],inputs['down'],inputs['downs'],inputs['lut'],layout=1,rows=a.rows,capacity=a.capacity,n_tile=a.n_tile,metadata_version=a.metadata_version)
        consumed=y+.03125
        metadata=[getattr(meta,n)for n in names]
        metadata_consumers=[v+17 for v in metadata]
    sync();print('capture_complete',flush=True)
    addresses=[v.data_ptr()for v in [y,consumed,*metadata,*metadata_consumers]]
    # Every route is active in the original mixed fixture, providing independent
    # expert partials for changed assignment/order without recompiling a graph.
    dots=None if diverse else np.load(a.fixture/'down_exact.npy')
    states=(*fixture['states'],'invalid','restored') if diverse else ('original','reverse','invalid','restored')
    for state in states:
        ids=cpu['ids'].clone();routing=cpu['routing'].clone()
        state_path=a.fixture/('uniform' if state in ('invalid','restored') else state)
        if diverse:
            ids=torch.from_numpy(np.load(state_path/'ids.npy'))
            routing=torch.from_numpy(np.load(state_path/'routing.npy'))
            inputs['x'].copy_(torch.from_numpy(np.load(state_path/'x.npy')).bfloat16())
        if state=='reverse':ids=ids.flip(1).contiguous();routing=-routing.flip(1).contiguous()
        if state=='invalid':ids[0,0]=-1
        inputs['ids'].copy_(ids);inputs['routing'].copy_(routing);sync()
        graph.replay(asynchronous=True);sync()
        actual=y.cpu();consumer=consumed.cpu();got=[v.cpu()for v in metadata+metadata_consumers]
        want=reference(ids.tolist(),fixture['E'],meta.rows,meta.capacity)
        meta_equal=all(torch.equal(got[i],torch.tensor(want[n],dtype=torch.int32))and torch.equal(got[i+len(names)],got[i]+17)for i,n in enumerate(names))
        assert addresses==[v.data_ptr()for v in [y,consumed,*metadata,*metadata_consumers]]
        check=dict(state=state,metadata_equal=meta_equal,status=want['status'][0])
        if want['status'][0]:
            ok=bool(actual.isnan().all() and consumer.isnan().all());check['all_failure_outputs_nan']=ok
        else:
            expected=np.load(state_path/'expected.npy') if diverse else np.zeros(actual.shape,np.float32)
            # Independent absolute-product tensor for original routing may have
            # cancellations removed; reversing/signing preserves this bound.
            absolute=np.load((state_path if diverse else a.fixture)/'absolute.npy')
            if not diverse:
                for t in range(fixture['T']):
                    for r in range(2):
                        e=int(ids[t,r]);expected[t]=(dots[t,e].astype(np.float32).astype(np.float64)*float(routing[t,r])+expected[t].astype(np.float64)).astype(np.float32)
            got_array=actual.numpy();delta=np.abs(got_array.astype(np.float64)-expected)
            u=2**-24;gamma=(256+fixture['R'])*u/(1-(256+fixture['R'])*u)
            limit=gamma*absolute+4*np.abs(np.spacing(expected)).astype(np.float64)
            count=int(np.count_nonzero(~np.isfinite(got_array)|(delta>limit)))
            consumer_equal=torch.equal(consumer.view(torch.uint8),(actual+.03125).view(torch.uint8))
            ok=count==0 and consumer_equal
            check.update(bad=count,max_abs=float(delta.max()),max_fp32_bound_ratio=float(np.max(delta/np.maximum(limit,np.finfo(np.float64).tiny))),consumer_bits_equal=consumer_equal)
        torch.save(dict(y=actual,consumer=consumer,metadata=dict(zip(names,got[:len(names)])),ids=ids,routing=routing),out/(state+'.pt'))
        result['checks'].append(check);save();assert meta_equal and ok,check
    # Same five complete-graph warmups in every independent process arm.
    for _ in range(5):graph.replay(asynchronous=True)
    sync();result['warmup_replays']=5;save()
    for trial in range(4):
        start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
        with torch.hpu.stream(stream):
            begin=time.perf_counter();start.record(stream)
            for _ in range(a.replays):graph.replay(asynchronous=True)
            torch.hpu.synchronize();end.record(stream);end.synchronize()
        result['timing'].append(dict(event_us=start.elapsed_time(end)*1000/a.replays,wall_us=(time.perf_counter()-begin)*1e6/a.replays));save()
    result.update(status='PASS_TP_LOCAL_MOE_FP32_BOUND',C=meta.rows,B=meta.capacity,
        event_median_us=statistics.median(v['event_us']for v in result['timing']),
        wall_median_us=statistics.median(v['wall_us']for v in result['timing']))
    save()
