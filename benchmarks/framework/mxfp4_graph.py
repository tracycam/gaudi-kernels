"""Static unequal groups in a single public-API HPU Graph, with placement dumps."""
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import time
from validation import timings_agree
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),
    GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),SRAM_SLICER_GRAPH_VISUALIZATION='1')
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.torch_linear import load_extension
from gaudi_kernels.mxfp4 import prepare_mxfp4,make_mxfp4_lut,grouped_mxfp4_static
load_extension(root/'artifacts/builds/torch-d/gaudi_kernels_torch.so')
torch.manual_seed(31);torch.set_num_threads(4)
counter=ctypes.CDLL(None).gaudi_kernel_launch_count;counter.restype=ctypes.c_ulonglong
lut=make_mxfp4_lut().to('hpu')
values=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64)


def sync():hc.mark_step();torch.hpu.synchronize()


records=[]
for n,k,ms in [(513,257,[1,0,3,17]),(512,6144,[1,0,3,17]),(6144,256,[1,0,3,17]),(512,6144,[257])]:
    inputs=[];weights=[];biases=[];fixtures=[];references=[];absolute=[]
    for m in ms:
        q=torch.randint(0,16,(n,k),dtype=torch.uint8)
        rows=torch.zeros(n,(k+1)//2,dtype=torch.uint8);rows[:]=q[:,0::2];rows[:,:k//2]|=q[:,1::2]<<4
        scales=torch.randint(118,133,(n,(k+31)//32),dtype=torch.uint8)
        packed=prepare_mxfp4(rows,scales,logical_k=k);x=torch.randn(m,k).bfloat16();b=torch.randn(n)*.1
        exact=values[q.long()]*torch.exp2((scales.double()-127).repeat_interleave(32,1)[:,:k])
        references.append(x.double()@exact.T+b.double())
        absolute.append(x.double().abs()@exact.abs().T+b.double().abs())
        inputs.append(x.to('hpu'));weights.append(packed.to('hpu'));biases.append(b.to('hpu'))
        fixtures.append({'rows':rows,'scales':scales,'input':x,'bias':b,'expected':references[-1]})
    sync();graph=torch.hpu.HPUGraph();stream=torch.hpu.Stream()
    with torch.hpu.graph(graph,stream=stream):ys=grouped_mxfp4_static(inputs,weights,lut,biases)
    sync();graph.replay();sync()
    errors=[];addresses=[]
    for i,(m,y,ref,ab) in enumerate(zip(ms,ys,references,absolute)):
        if not m:assert y is None;continue
        actual=y.cpu().double();delta=(actual-ref).abs()
        assert torch.isfinite(actual).all() and bool((delta<=2e-5+2e-6*ab).all()),(n,k,m,delta.max())
        errors.append(float(delta.norm()/ref.norm()));addresses.append(y.data_ptr());fixtures[i]['actual']=actual
    before=counter()
    for _ in range(10):graph.replay(asynchronous=True)
    sync();launches=counter()-before;assert launches==10,launches
    event=[];wall=[]
    for _ in range(5):
        begin,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
        t=time.perf_counter();begin.record(stream)
        for _ in range(20):graph.replay(asynchronous=True)
        sync();end.record(stream);end.synchronize();sync()
        event.append(begin.elapsed_time(end)*1000/20);wall.append((time.perf_counter()-t)*1e6/20)
    assert timings_agree(event,wall),(event,wall)
    for x,f in zip(inputs,fixtures):
        if x.shape[0]:x.copy_(-f['input'])
    sync();graph.replay();sync()
    changed=[]
    for m,y,f,ab in zip(ms,ys,fixtures,absolute):
        if not m:continue
        ref=-f['expected']+2*f['bias'].double();actual=y.cpu().double()
        assert bool(((actual-ref).abs()<=2e-5+2e-6*ab).all());changed.append(y.data_ptr())
        f['changed']=actual
    assert addresses==changed
    rec={'N':n,'K':k,'M_groups':ms,'full_output_elements':sum(ms)*n,'relative_l2_per_group':errors,
         'synapse_launches_for10_replays':launches,'event_us':event,'wall_us':wall,
         'median_event_us':statistics.median(event),'median_wall_us':statistics.median(wall),
         'changed_inputs_pass':True,'stable_output_addresses':True,'numerical_pass':True,
         'placement_pass':'requires_compiled_audit','scope':'host-known groups; no device-dynamic routing'}
    records.append(rec);print(json.dumps(rec),flush=True)
    torch.save(fixtures,out/f'fixture-N{n}-K{k}-M{sum(ms)}.pt')
(out/'result.json').write_text(json.dumps({'status':'NUMERICAL_AND_REPLAY_PASS','records':records},indent=2)+'\n')
