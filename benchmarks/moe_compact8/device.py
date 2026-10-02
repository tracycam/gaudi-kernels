"""Production broadcast versus compact folded-GP/vector-down, full TP-local MoE.

Uses frozen existing DLLs without recompilation/private boundary rewrites. Run
only under the standard exclusive module runner after the parent releases cards.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--runtime-fixture',type=Path,required=True)
p.add_argument('--weights-fixture',type=Path,required=True);p.add_argument('--replays',type=int,default=16)
p.add_argument('--profile-only',action='store_true');a=p.parse_args()
out=Path(os.environ['PROBE_OUT']);runtime=a.runtime_fixture.resolve(strict=True);fixture=a.weights_fixture.resolve(strict=True)
if os.environ.get('GAUDI_KERNELS_MODULE_ID')!='6' or os.environ.get('HABANA_VISIBLE_MODULES')!='6':raise RuntimeError('module6 runner required')
plan=json.loads((runtime/'plan.json').read_text());manifest=json.loads((fixture/'fixture.json').read_text())
assert plan['format']=='gk-moe-compact8-runtime-v1' and manifest['status']=='IMMUTABLE_FIXTURE_READY' and manifest['experts']==384
for name,record in plan['files'].items():assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==record['sha256'],name
registered={str(Path(x).resolve())for x in os.getenv('GC_KERNEL_PATH','').split(':')if x}
assert all(str((runtime/name).resolve())in registered for name in plan['tpc_libraries'])
os.environ.update(PT_HPU_LAZY_MODE='1',PT_ENABLE_INT64_SUPPORT='0',NATIVE_BRIDGE='pt2',
    UNIFIED_PRECISION_ROUTER='1',UNIFIED_PRECISION_GATE='1',GK_MXFP4_GP_ENABLED='0',
    GK_MXFP4_FOLDED_ENABLED='1',GK_MXFP4_DOWN_ENABLED='1',
    GK_MXFP4_FOLDED_TORCH_LIBRARY=str(runtime/'libraries/folded/gaudi_moe_activation_folded.so'),
    GK_MXFP4_DOWN_TORCH_LIBRARY=str(runtime/'libraries/down/gaudi_down_activation.so'),
    ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),
    GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'))
(out/'graphs').mkdir();(out/'runtime-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(out/'weight-fixture.json').write_text(json.dumps(manifest,indent=2)+'\n')
import numpy as np
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0,str(runtime/'executor'))
import native_ops
import batch_ops
from ledger import accounting
torch.set_num_threads(2)
batch_ops.set_gp_policy('vector_fetch_folded');batch_ops.set_down_policy('vector_fetch')

def sync():hc.mark_step();torch.hpu.synchronize()
def same(x,y):return torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
def graph_ids():
    p=out/'post_graph.json';paths=[p]if p.is_file()else list(p.rglob('*.json'))if p.is_dir()else[]
    return {g['name'] for f in paths for g in json.loads(f.read_text())['graphs']}
def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

# Persistent uint8 owners only. File hash validation/loading is outside timing.
owners=[];mapped=[]
for name in ['gp','gs','dp','ds']:
    r=manifest['weights'][name];path=fixture/(name+'.bin');assert path.stat().st_size==r['bytes'] and sha_file(path)==r['sha256']
    data=np.memmap(path,dtype=np.uint8,mode='c',shape=tuple(r['shape']));mapped.append(data)
    owners.append(torch.from_numpy(data).to('hpu'))
gp,gs,dp,ds=owners;table,directions=native_ops.constants('hpu');sync()
result={'status':'CHECKING','source_contract':'E384 R8 GP512x6144 down6144x256 historical N512 original-width storage',
        'checks':[],'graphs':[],'timing':[],'scope':'complete TP-local GP/SiLU/down/ordered FP32 route combine; no TP collective/model TPS claim',
        'profile_only':a.profile_only,'torch':torch.__version__}
graphs={};inputs={};outputs={};streams={};addresses={}
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
def load_case(case):
    path=fixture/case['file'];assert sha_file(path)==case['sha256'];return torch.load(path,map_location='cpu',weights_only=True)
def update(tokens,raw):
    for name in ('x','ids','routing'):inputs[tokens][name].copy_(raw[name])
    sync()

with torch.inference_mode():
    for tokens in (2,8):
        case=next(c for c in manifest['cases']if c['tokens']==tokens and c['state']=='sparse');raw=load_case(case)
        inputs[tokens]={name:value.to('hpu')for name,value in raw.items()};sync()
        for mode in ('broadcast','compact'):
            key=(tokens,mode);before=graph_ids();stream,graph=torch.hpu.Stream(),torch.hpu.HPUGraph()
            with torch.hpu.graph(graph,stream=stream):
                produced=inputs[tokens]['x']+.015625
                y=batch_ops.moe(produced,inputs[tokens]['ids'],inputs[tokens]['routing'],gp,gs,dp,ds,table,directions,mode=mode)
                consumed=y+.03125
            assert tuple(y.shape)==(tokens,6144) and y.dtype==torch.float32
            sync();graphs[key]=graph;streams[key]=stream;outputs[key]=(y,consumed);addresses[key]=[t.data_ptr()for t in outputs[key]]
            graph.replay(asynchronous=True);sync()
            result['graphs'].append({'tokens':tokens,'variant':mode,'postgraph_names':sorted(graph_ids()-before)})
    # Every full-output state passes before any timed replay.
    for case in manifest['cases']:
        tokens=case['tokens'];raw=load_case(case);update(tokens,raw);saved=dict(raw);check={'case':case['name'],'variants':{}}
        result['checks'].append(check)
        for mode in ('broadcast','compact'):
            key=(tokens,mode);graphs[key].replay(asynchronous=True);sync()
            actual=[t.cpu()for t in outputs[key]];saved[mode]={'moe_fp32':actual[0],'consumer_fp32':actual[1]}
            assert [t.data_ptr()for t in outputs[key]]==addresses[key]
            finite=all(bool(t.isfinite().all())for t in actual)
            equal=all(same(t,saved['broadcast'][field])for t,field in zip(actual,('moe_fp32','consumer_fp32')))
            check['variants'][mode]={'finite':finite,'all_fp32_bits_equal':equal,'words_checked':sum(t.numel()for t in actual)}
            torch.save(saved,out/(case['name']+'.pt'));save();assert finite and equal,(case,mode,check)
            for _ in range(3):graphs[key].replay(asynchronous=True)
            sync();assert all(same(t.cpu(),actual[i])for i,t in enumerate(outputs[key]))
        ledger=accounting(raw['ids'].tolist());(out/(case['name']+'-ledger.json')).write_text(json.dumps(ledger,indent=2)+'\n')
        save();print(json.dumps(check),flush=True)
    result['status']='PASS_FULL_OUTPUT_BITS'
    for case in manifest['cases']:
        if case['state']not in ('hot','sparse','rotated'):continue
        tokens=case['tokens'];raw=load_case(case);update(tokens,raw)
        for trial in range(1 if a.profile_only else 3):
            for mode in ('broadcast','compact','compact','broadcast'):
                key=(tokens,mode);graph,stream=graphs[key],streams[key]
                count=3 if a.profile_only else a.replays
                start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
                with torch.hpu.stream(stream):
                    begin=time.perf_counter();start.record(stream)
                    for _ in range(count):graph.replay(asynchronous=True)
                    # Drain the bridge's async host submission before queuing
                    # the final capture-stream event (previous 4us bug avoided).
                    torch.hpu.synchronize();end.record(stream);end.synchronize()
                wall=(time.perf_counter()-begin)*1e6/count;event=start.elapsed_time(end)*1000/count
                r={'case':case['name'],'trial':trial,'variant':mode,'event_us':event,'wall_us':wall,'replays':count}
                result['timing'].append(r);save()
                if not a.profile_only:assert .5<event/wall<1.2,r
    result['status']='PASS_FULL_MOE_PAIRED';result['dispatch']={'gp':batch_ops.gp_snapshot(),'down':batch_ops.down_snapshot()};save()
