"""Bounded public block FP8 graph correctness, changed-input replay and timing.

Run exclusively inside run_device_probe. No independent device acquisition.
Set GK_BLOCK_BUILD to the isolated TPC/Torch build directory before launching.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--cases',required=True,type=Path);a=p.parse_args()
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',
                  GRAPH_VISUALIZATION_DIR=str(out/'graphs'),DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.block_fp8 import prepare_block_fp8,linear_block_fp8,linear_block_fp8_quantized
from oracle import reference,errors,quantize_block
torch.ops.load_library(str(root/os.environ['GK_BLOCK_BUILD']/'gaudi_block_fp8_torch.so'))
torch.set_num_threads(4);torch.manual_seed(280901)
def sync():hc.mark_step();torch.hpu.synchronize()
records=[]
for case in json.loads(a.cases.read_text()):
    name=case['name'];dest=out/name;dest.mkdir();m,n,k=case['shape'];depth=case.get('depth',1)
    act=case['activation'];sm=case.get('scale_math','fp32')
    x0=torch.randn(m,k).bfloat16()
    fixture=Path(case['fixture']) if 'fixture' in case else None
    if fixture is not None:
        assert (n,k)==(3392,6144) and depth==1
        x0=torch.frombuffer(bytearray((fixture/'activation.bin').read_bytes()),dtype=torch.bfloat16).reshape(-1,k)[:m].clone()
        assert x0.shape[0]==m
    if case.get('input_fixture')=='zero':
        x0.zero_();x0[:,1::2]=-0.0
    if case.get('input_fixture')=='cancellation':
        x0.fill_(1);x0[:,1::2]=-1
    x1=(torch.ones_like(x0) if case.get('input_fixture') in ('zero','cancellation') else -x0)
    cpu=[];saved={'case':case,'x':x0,'layers':[]}
    for i in range(depth):
        width=k if i==0 else n
        raw=(torch.randn(n,width)*64).clamp(-448,448).to(torch.float8_e4m3fn)
        # Include zero, tiny subnormal, highest codes, and signed cancellation.
        raw.view(torch.uint8).reshape(-1)[:16]=torch.tensor([0,128,1,129,7,135,15,143,119,247,120,248,126,254,56,184],dtype=torch.uint8)
        scales=torch.rand((n+127)//128,(width+127)//128)*.0002+.0001
        bias=torch.randn(n)*.01
        if fixture is not None:
            raw=torch.frombuffer(bytearray((fixture/'weight.bin').read_bytes()),dtype=torch.uint8).reshape(n,k).clone().view(torch.float8_e4m3fn)
            scales=torch.frombuffer(bytearray((fixture/'scales.bin').read_bytes()),dtype=torch.float32).reshape((n+127)//128,(k+127)//128).clone()
            bias=torch.zeros(n)
        if case.get('input_fixture')=='cancellation':
            pairs=width//2;raw.view(torch.uint8)[:,1:2*pairs:2]=raw.view(torch.uint8)[:,:2*pairs:2]
            if width%2:raw.view(torch.uint8)[:,-1]=0
        prep=prepare_block_fp8(raw,scales,bias);cpu.append((raw,scales,bias,prep))
        saved['layers'].append({'raw':raw,'scales':scales,'bias':bias,'prepared_bytes':prep.weight.view(torch.uint8),'prepared_scales':prep.scales,
            'checkpoint_weight_sha256':prep.checkpoint_weight_sha256,'checkpoint_scales_sha256':prep.checkpoint_scales_sha256,'rounded_half_values':prep.rounded_half_values})
    torch.save(saved,dest/'inputs-outputs.pt')
    x=x0.to('hpu');weights=[c[-1].to('hpu') for c in cpu];sync()
    if case.get('quantized_input'):
        assert act=='per_block_fp8' and depth==1
        q0,s0,_,_=quantize_block(x0);qx=q0.to('hpu');qs=s0.to('hpu');sync()
    # Byte gate independent of full-MAC acceptance, on actual compiled quant.
    if case.get('quant_gate',False):
        q,sa=torch.ops.gaudi_block_fp8.quant(x);sync();qa=q.cpu();saa=sa.cpu();expected,es,_,_=quantize_block(x0)
        saved['quant']={'actual_bytes':qa.view(torch.uint8),'expected_bytes':expected.view(torch.uint8),'actual_scales':saa,'expected_scales':es}
        torch.save(saved,dest/'inputs-outputs.pt')
        mismatches=int((qa.view(torch.uint8)!=expected.view(torch.uint8)).sum())
        assert mismatches==0 and torch.equal(saa,es),(name,'quant mismatch',mismatches)
        del q,sa
    stream,graph=torch.hpu.Stream(),torch.hpu.HPUGraph()
    with torch.hpu.graph(graph,stream=stream):
        y=x
        if case.get('quantized_input'):
            y=linear_block_fp8_quantized(qx,qs,weights[0])
        else:
            for w in weights:y=linear_block_fp8(y,w,activation=act,scale_math=sm,row_tile=case.get('row_tile'))
    sync();graph.replay(asynchronous=True);sync();actual=y.cpu()
    def check(source,actual):
        refs=None
        for raw,s,b,prep in cpu:
            refs=reference(source,raw,s,b,prep,act,sm)
            source=refs['adapted_fp64'].bfloat16()
        e=errors(actual,refs)
        assert e['finite'] and e['adapted_fp64']['relative_l2']<.006,(name,e)
        # The A8 error allowance follows that fixture's GPU-style quantization
        # error, not one universal loose bound concealing preprocessing loss.
        gpu_error=errors(refs['gpu_style_fp64'].bfloat16(),refs)['high_fp64']['relative_l2']
        assert e['high_fp64']['relative_l2']<=1.15*gpu_error+.008,(name,e,gpu_error)
        return e,refs
    initial,refs=check(x0,actual);saved.update(actual=actual,oracles=refs)
    x.copy_(x1)
    if case.get('quantized_input'):
        q1,s1,_,_=quantize_block(x1);qx.copy_(q1);qs.copy_(s1)
    sync();graph.replay(asynchronous=True);sync();changed=y.cpu()
    changed_error,changed_refs=check(x1,changed);assert not torch.equal(actual,changed)
    saved.update(changed=changed,changed_oracles=changed_refs);torch.save(saved,dest/'inputs-outputs.pt')
    for _ in range(5):graph.replay(asynchronous=True)
    sync();events=[];walls=[];repeats=case.get('repeats',30)
    for sample in range(3):
        begin,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
        start=time.perf_counter();begin.record(stream)
        for _ in range(repeats):graph.replay(asynchronous=True)
        # Drain the asynchronous bridge submission worker before recording the
        # end marker; otherwise it can overtake queued replay requests and
        # report a misleading ~4us device interval against ~38us wall time.
        sync();end.record(stream);end.synchronize();sync()
        events.append(begin.elapsed_time(end)*1000/repeats);walls.append((time.perf_counter()-start)*1e6/repeats)
    assert max(statistics.median(events),statistics.median(walls))/min(statistics.median(events),statistics.median(walls))<1.25,(name,events,walls)
    row={'case':case,'status':'PASS_NUMERICAL_PLACEMENT_UNAUDITED','checked_outputs':2*actual.numel(),
         'initial_errors':initial,'changed_errors':changed_error,'event_us':events,'wall_us':walls,
         'median_event_us':statistics.median(events),'median_wall_us':statistics.median(walls),
         'native_replay':False,'retained_intermediate_outputs':False}
    records.append(row);(dest/'result.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
    (out/'result.json').write_text(json.dumps({'status':'INCOMPLETE','records':records},indent=2)+'\n')
    del graph,weights,x,y
    sync()
(out/'result.json').write_text(json.dumps({'status':'PASS_NUMERICAL_PLACEMENT_UNAUDITED','records':records},indent=2)+'\n')
