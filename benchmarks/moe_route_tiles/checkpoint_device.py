"""Replay saved real checkpoint inputs through old and routed public graphs.

This captures numerical differences for investigation. It does not turn a
cross-algorithm comparison into an independent numerical or model quality gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

p=argparse.ArgumentParser()
for name in ('input','runtime-fixture','core-library','metadata-library','tiles-library'):
    p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--input-sha256',required=True)
p.add_argument('--rows',type=int,choices=(8,16,32),default=32)
p.add_argument('--timing-trials',type=int,default=0)
p.add_argument('--replays',type=int,default=8)
a=p.parse_args()
assert 0<=a.timing_trials<=5 and 1<=a.replays<=16
assert hashlib.sha256(a.input.read_bytes()).hexdigest()==a.input_sha256
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
assert os.environ['HABANA_VISIBLE_MODULES']==os.environ['GAUDI_KERNELS_MODULE_ID']
os.environ.update(PT_HPU_LAZY_MODE='1',PT_ENABLE_INT64_SUPPORT='0',UNIFIED_PRECISION_ROUTER='1',
    GK_MXFP4_GP_ENABLED='0',GK_MXFP4_FOLDED_ENABLED='0',GK_MXFP4_SCALE_TAIL_ENABLED='0',GK_MXFP4_DOWN_ENABLED='0',
    ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),
    DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
runtime=a.runtime_fixture.resolve();plan=json.loads((runtime/'plan.json').read_text())
assert plan['format']=='gk-moe-compact8-runtime-v1'
for name,r in plan['files'].items():assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==r['sha256']
sys.path.insert(0,str(runtime/'executor'));import native_ops,batch_ops,precision_ops
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'moe_route_metadata_v3'))
from reference import reference
from gaudi_kernels.mxfp4_route_tiles import moe_route_tiles
for f in (a.core_library,a.metadata_library,a.tiles_library):torch.ops.load_library(str(f.resolve()))
# This is our own checkpoint diagnostic tensor file, pinned above before pickle.
cpu=torch.load(a.input,weights_only=False,map_location='cpu')
assert cpu['x'].shape==(512,6144) and cpu['ids'].shape==(512,8)
assert cpu['ids'].dtype==torch.int32 and cpu['x'].dtype==torch.bfloat16
for key in ('x','routing'):assert torch.isfinite(cpu[key]).all()
for key in ('gp','gs','dp','ds'):assert cpu[key].dtype==torch.uint8
inputs={k:v.to('hpu')for k,v in cpu.items()}
torch.set_num_threads(2)
def sync():hc.mark_step();torch.hpu.synchronize()
names=('counts','row_status','prefix','tile_expert','tile_base','valid_rows','status','row_map','inverse','chunk_offsets')
result=dict(status='CAPTURING',input_sha256=a.input_sha256,checks=[],rows=a.rows,timing=[],
            scope='saved real checkpoint TP rank0 inputs; same-input diagnostics only, no model acceptance or TPS')
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
with torch.inference_mode():
    sync();graphs={};outputs={};metadatas={};streams={}
    for mode in ('broadcast','routed_pure','routed_metadata'):
        graph=torch.hpu.HPUGraph();stream=torch.hpu.Stream()
        with torch.hpu.graph(graph,stream=stream):
            x=inputs['x']+0
            if mode=='broadcast':
                y=batch_ops.moe(x,inputs['ids'],inputs['routing'],inputs['gp'],inputs['gs'],
                    inputs['dp'],inputs['ds'],inputs['table'],inputs['directions'],mode='broadcast')
            else:
                y,meta=moe_route_tiles(x,inputs['ids'],inputs['routing'],inputs['gp'],inputs['gs'],
                    inputs['dp'],inputs['ds'],inputs['table'],layout=1,rows=a.rows,n_tile=2048,metadata_version=3)
                if mode=='routed_metadata':metadatas[mode]=[getattr(meta,n)+0 for n in names]
            consumer=y+.03125
        sync();graphs[mode]=graph;outputs[mode]=(y,consumer);streams[mode]=stream
        print('CAPTURED',mode,flush=True)
    for state in ('original','route_reverse','restored'):
        ids=cpu['ids'].flip(1).contiguous()if state=='route_reverse'else cpu['ids']
        routing=cpu['routing'].flip(1).contiguous()if state=='route_reverse'else cpu['routing']
        inputs['ids'].copy_(ids);inputs['routing'].copy_(routing);sync();actual={}
        for mode,graph in graphs.items():
            graph.replay(asynchronous=True);sync()
            y,c=[v.cpu()for v in outputs[mode]];actual[mode]=y
            record=dict(y=y,consumer=c)
            check=dict(state=state,mode=mode,cells=y.numel(),finite=int(torch.isfinite(y).sum()),
                       consumer_bits_equal=torch.equal(c.view(torch.uint8),(y+.03125).view(torch.uint8)))
            if mode in metadatas:
                got=[v.cpu()for v in metadatas[mode]];want=reference(ids.tolist(),384,meta.rows,meta.capacity)
                record['metadata']=dict(zip(names,got))
                check['metadata_equal']=all(torch.equal(v,torch.tensor(want[n],dtype=torch.int32))for n,v in zip(names,got))
                check['metadata_status']=got[names.index('status')].tolist()
                assert check['metadata_equal'] and check['metadata_status']==[0],check
            torch.save(record,out/(state+'-'+mode+'.pt'));result['checks'].append(check);save()
            assert check['consumer_bits_equal']
        for mode in ('routed_pure','routed_metadata'):
            x,y=actual[mode],actual['broadcast'];delta=x-y
            result['checks'].append(dict(state=state,mode=mode,reference='broadcast',
                relative_l2=float(delta.norm()/y.norm()),max_abs=float(delta.abs().max()),
                reference_norm=float(y.norm()),candidate_norm=float(x.norm()),
                bit_differences=int((x.view(torch.int32)!=y.view(torch.int32)).sum())))
        result['checks'].append(dict(state=state,pure_debug_bitwise=torch.equal(
            actual['routed_pure'].view(torch.uint8),actual['routed_metadata'].view(torch.uint8))))
        save()
    # Timing is opt-in and retains the unmodified real checkpoint distribution.
    # Same resident owners, full producer/MoE/consumer graph, paired ABBA order.
    # Debug metadata output is not timed; no independent arithmetic claim.
    for trial in range(a.timing_trials):
        for arm,mode in enumerate(('broadcast','routed_pure','routed_pure','broadcast')):
            graph,stream=graphs[mode],streams[mode]
            with torch.hpu.stream(stream):
                for _ in range(5):graph.replay(asynchronous=True)
                sync()
                start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
                begin=time.perf_counter();start.record(stream)
                for _ in range(a.replays):graph.replay(asynchronous=True)
                end.record(stream);end.synchronize()
                wall=(time.perf_counter()-begin)*1e6/a.replays
            values=[v.cpu()for v in outputs[mode]]
            expected=actual[mode]
            same=torch.equal(values[0].view(torch.uint8),expected.view(torch.uint8))
            consumer_ok=torch.equal(values[1].view(torch.uint8),(expected+.03125).view(torch.uint8))
            path=out/f'timing-{trial}-{arm}-{mode}.pt'
            torch.save(dict(y=values[0],consumer=values[1]),path)
            result['timing'].append(dict(trial=trial,arm=arm,mode=mode,replays=a.replays,
                event_us=start.elapsed_time(end)*1000/a.replays,wall_us=wall,
                restored_output_bits_equal=same,consumer_bits_equal=consumer_ok,
                tensor_path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            save();assert same and consumer_ok
    result['timing_summary']={mode:dict(
        event_median_us=statistics.median(r['event_us']for r in result['timing']if r['mode']==mode),
        wall_median_us=statistics.median(r['wall_us']for r in result['timing']if r['mode']==mode))
        for mode in ('broadcast','routed_pure') if a.timing_trials}
    result['status']='COMPLETE_CHECKPOINT_DIAGNOSTIC_NOT_MODEL_ACCEPTANCE';save()
