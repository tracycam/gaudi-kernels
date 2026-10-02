"""Same-process v2/v3 metadata ABBA; exclusive module runner required."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

p=argparse.ArgumentParser()
for name in ('v2-library','v3-library'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--tokens',type=int,required=True);p.add_argument('--routes',type=int,default=8)
p.add_argument('--experts',type=int,default=384);p.add_argument('--rows',type=int,default=16)
p.add_argument('--replays',type=int,default=4);a=p.parse_args()
assert 1<=a.replays<=16
module=os.environ.get('GAUDI_KERNELS_MODULE_ID');assert module is not None and os.environ.get('HABANA_VISIBLE_MODULES')==module
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(PT_HPU_LAZY_MODE='1',ENABLE_EXPERIMENTAL_FLAGS='true',
    DUMP_POST_GRAPHS=str(out/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.route_metadata import geometry
from gaudi_kernels.route_metadata_v2 import route_metadata_v2
from gaudi_kernels.route_metadata_v3 import route_metadata_v3
from reference import reference
torch.set_num_threads(2)
for path in (a.v2_library,a.v3_library):torch.ops.load_library(str(path.resolve()))
c,b=geometry(a.tokens,a.routes,a.experts,a.rows)
names=('counts','row_status','prefix','tile_expert','tile_base','valid_rows','status','row_map','inverse')
cases={}
for mode in ('uniform','hot','skew','bad_id','negative','duplicate'):
    if mode=='duplicate'and a.routes==1:continue
    ids=torch.tensor([[(t*a.routes+r)%a.experts for r in range(a.routes)]for t in range(a.tokens)],dtype=torch.int32)
    if mode in ('hot','skew'):ids[:a.tokens if mode=='hot'else 3*a.tokens//4]=torch.arange(a.experts-a.routes,a.experts,dtype=torch.int32).flip(0)
    if mode=='bad_id':ids[0,0]=a.experts
    if mode=='negative':ids[0,0]=-1
    if mode=='duplicate':ids[0,1]=ids[0,0]
    cases[mode]=ids;torch.save(ids,out/(mode+'-ids.pt'))
result=dict(status='CHECKING',T=a.tokens,R=a.routes,E=a.experts,C=c,B=b,checks=[],timings=[],
    libraries={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()for p in (a.v2_library,a.v3_library)},
    scope='Metadata producer and identical nine-output consumers; no model/MME or physical bandwidth claim')
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
def sync():hc.mark_step();torch.hpu.synchronize()
graphs={};streams={};outputs={};owners={}
with torch.inference_mode():
    source=cases['uniform'].to('hpu');sync()
    for variant,fn in [('v2',route_metadata_v2),('v3',route_metadata_v3)]:
        graph=torch.hpu.HPUGraph();stream=torch.hpu.Stream()
        with torch.hpu.graph(graph,stream=stream):
            metadata=fn(source+0,experts=a.experts,rows=c,capacity=b)
            actual=[getattr(metadata,n)for n in names]
            consumed=[v+17 for v in actual]
            del metadata  # only the same nine public fields escape either capture
        sync();graphs[variant]=graph;streams[variant]=stream;outputs[variant]=actual+consumed
        owners[variant]=[v.data_ptr()for v in outputs[variant]]
    for state,ids in cases.items():
        source.copy_(ids);sync();want=reference(ids.tolist(),a.experts,c,b)
        record=dict(state=state,status=want['status'][0],variants={})
        for variant in ('v2','v3'):
            graphs[variant].replay(asynchronous=True);sync();cpu=[v.cpu()for v in outputs[variant]]
            assert owners[variant]==[v.data_ptr()for v in outputs[variant]]
            same=all(torch.equal(cpu[i],torch.tensor(want[n],dtype=torch.int32))and torch.equal(cpu[i+9],cpu[i]+17)for i,n in enumerate(names))
            record['variants'][variant]=dict(all_nine_outputs_and_consumers_equal=same)
            torch.save(dict(zip([*names,*[n+'_consumer'for n in names]],cpu)),out/(state+'-'+variant+'.pt'))
            result['checks'].append(dict(state=state,variant=variant,equal=same));save();assert same,record
    for state in ('uniform','hot','skew'):
        source.copy_(cases[state]);sync()
        for variant in ('v2','v3'):graphs[variant].replay(asynchronous=True);sync()
        for trial in range(2):
            for variant in ('v2','v3','v3','v2'):
                stream=streams[variant];start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
                with torch.hpu.stream(stream):
                    begin=time.perf_counter();start.record(stream)
                    for _ in range(a.replays):graphs[variant].replay(asynchronous=True)
                    torch.hpu.synchronize();end.record(stream);end.synchronize()
                event=start.elapsed_time(end)*1000/a.replays;wall=(time.perf_counter()-begin)*1e6/a.replays
                result['timings'].append(dict(state=state,trial=trial,variant=variant,event_us=event,wall_us=wall,replays=a.replays));save()
                assert event>0 and event<wall*1.2 and wall<max(event*1.5,event+30),(event,wall)
    result['status']='PASS_METADATA_PAIRED';save()
