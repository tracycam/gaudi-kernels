"""Experimental large-M broadcast versus unchanged scale-tail/vector-down ELFs.

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
p.add_argument('--weights-fixture',type=Path,required=True);p.add_argument('--replays',type=int,default=4)
p.add_argument('--profile-only',action='store_true')
p.add_argument('--compact-kind',choices=('vector','scalar'),default='vector')
p.add_argument('--tokens',type=int,choices=(32,128,512,513),required=True)
p.add_argument('--scale-torch',type=Path,required=True);p.add_argument('--scale-tpc',type=Path,required=True)
a=p.parse_args();assert 1<=a.replays<=16
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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'moe_compact8'))
from ledger import accounting
torch.set_num_threads(2)
batch_ops.set_gp_policy('vector_fetch_folded');batch_ops.set_down_policy('vector_fetch')
scale_pins={'torch':'c73a48e86e32fb5b531cc97d5c205bc35f28c654820773cd82814702688db410',
            'tpc':'f45f038808d6ec6ba14ac99445fe7bdce35a523c6f63271a3f37624e57130e07'}
for kind,path in [('torch',a.scale_torch),('tpc',a.scale_tpc)]:
    assert hashlib.sha256(path.read_bytes()).hexdigest()==scale_pins[kind]
assert str(a.scale_tpc.resolve())in registered
assert os.environ.get('GK_MXFP4_SCALE_TAIL_ENABLED','0')=='0','Do not bypass production qualification guards'
torch.ops.load_library(str(a.scale_torch))
from precision_ops import combine

def experimental_compact(x,ids,routing):
    # Independent probe only. Original production rows<=8 guard is unchanged.
    x=x.clone();ids=ids.to(torch.int32).clone();routing=routing.float().clone()
    gp_call=torch.ops.gaudi_gp_scale_tail.gp if a.compact_kind=='vector' else torch.ops.unified_batch.gp
    down_call=torch.ops.gaudi_down_activation.broadcast if a.compact_kind=='vector' else torch.ops.unified_batch.down
    p=gp_call(gp,gs,x,table,ids)
    g=torch.ops.unified_batch.gate(p,ids)
    d=down_call(dp,ds,g,table,ids)
    return combine(d,routing,directions,6144,8)


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
result={'status':'CHECKING','experimental_rows':a.tokens,'production_guard_changed':False,'scale_library_sha256':scale_pins,'source_contract':'E384 R8 GP512x6144 down6144x256 historical N512 original-width storage',
        'checks':[],'graphs':[],'timing':[],'scope':'complete TP-local GP/SiLU/down/ordered FP32 route combine; no TP collective/model TPS claim',
        'profile_only':a.profile_only,'compact_kind':a.compact_kind,'torch':torch.__version__}
graphs={};inputs={};outputs={};streams={};addresses={}
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
torch.manual_seed(928513)
x_cpu=(torch.randn(a.tokens,6144)*.1).bfloat16()
cases=[];case_inputs={}
for state in ('uniform','hot','skew','zero'):
    ids=torch.tensor([[(row*8+slot)%384 for slot in range(8)]for row in range(a.tokens)],dtype=torch.int32)
    if state in ('hot','skew'):
        count=a.tokens if state=='hot'else 3*a.tokens//4
        ids[:count]=torch.arange(376,384,dtype=torch.int32)
    routing=torch.rand(a.tokens,8,dtype=torch.float32);routing/=routing.sum(-1,keepdim=True)
    if state=='zero':routing.zero_()
    name=f't{a.tokens}-{state}'
    raw={'x':x_cpu if state in ('uniform','zero')else -x_cpu.roll(17,-1),'ids':ids,'routing':routing}
    torch.save(raw,out/(name+'-inputs.pt'));case_inputs[name]=raw
    cases.append({'name':name,'tokens':a.tokens,'state':state})
def load_case(case):return case_inputs[case['name']]
def update(tokens,raw):
    for name in ('x','ids','routing'):inputs[tokens][name].copy_(raw[name])
    sync()

with torch.inference_mode():
    for tokens in (a.tokens,):
        case=next(c for c in cases if c['tokens']==tokens and c['state']=='uniform');raw=load_case(case)
        inputs[tokens]={name:value.to('hpu')for name,value in raw.items()};sync()
        for mode in ('broadcast','compact'):
            key=(tokens,mode);before=graph_ids();stream,graph=torch.hpu.Stream(),torch.hpu.HPUGraph()
            with torch.hpu.graph(graph,stream=stream):
                produced=inputs[tokens]['x']+.015625
                y=(batch_ops.moe(produced,inputs[tokens]['ids'],inputs[tokens]['routing'],gp,gs,dp,ds,table,directions,mode='broadcast')
                   if mode=='broadcast'else experimental_compact(produced,inputs[tokens]['ids'],inputs[tokens]['routing']))
                consumed=y+.03125
            assert tuple(y.shape)==(tokens,6144) and y.dtype==torch.float32
            sync();graphs[key]=graph;streams[key]=stream;outputs[key]=(y,consumed);addresses[key]=[t.data_ptr()for t in outputs[key]]
            graph.replay(asynchronous=True);sync()
            result['graphs'].append({'tokens':tokens,'variant':mode,'postgraph_names':sorted(graph_ids()-before)})
    # Every full-output state passes before any timed replay.
    for case in cases:
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
        ledger=accounting(raw['ids'].tolist());ledger['interval_count']=len(ledger.pop('intervals')); (out/(case['name']+'-ledger.json')).write_text(json.dumps(ledger,indent=2)+'\n')
        save();print(json.dumps(check),flush=True)
    result['status']='PASS_FULL_OUTPUT_BITS'
    for case in cases:
        if case['state']not in ('hot','uniform','skew'):continue
        tokens=case['tokens'];raw=load_case(case);update(tokens,raw)
        for trial in range(1 if a.profile_only else 2):
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
    result['status']='PASS_FULL_MOE_PAIRED';result['dispatch']={'candidate':'experimental direct four-node composition with pinned existing ELFs','baseline':'frozen production broadcast','production_guard_changed':False};save()
