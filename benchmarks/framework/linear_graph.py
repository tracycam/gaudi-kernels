"""Two consecutive linears: full outputs, changed-input replay, correct-stream timing."""
import hashlib
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import time
from validation import timings_agree

import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'python'))
from gaudi_kernels.torch_linear import load_extension,linear_bf16,linear_fp8_w8a8_prepared
torch.set_num_threads(4);torch.manual_seed(260926)
out=Path(os.environ['PROBE_OUT'])
asynchronous=os.environ.get('GK_GRAPH_ASYNC','0')=='1'
library=root/os.environ.get('GK_TORCH_BUILD','artifacts/builds/torch-c')/'gaudi_kernels_torch.so'
load_extension(library)


def sync():
    hc.mark_step();torch.hpu.synchronize()


def reference(x,w,b,fp8):
    if fp8:
        scale=x.float().abs().amax(-1,keepdim=True).clamp_min(1e-10)*(1/448)
        q=(x.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()
        # Native activation adaptation halves OCP values; RNE at subnormal edge.
        q=(q*.5).to(torch.float8_e4m3fn).double()*(scale.double()*2)
    else:q=x.double()
    return (q@w.double().T+b.double()).bfloat16()


records=[];saved={}
counter=ctypes.CDLL(None)
count=getattr(counter,'gaudi_kernel_launch_count',None)
if count:count.restype=ctypes.c_ulonglong
for fp8 in [False,True]:
    for m,n,k in [(1,129,257),(16,128,256),(513,128,256)]:
        x0=torch.randn(m,k).bfloat16();w0=(torch.randn(n,k)*16).bfloat16()
        if fp8:w0=w0.float().clamp(-240,240).to(torch.float8_e4m3fn)
        w1=(torch.randn(67,n)*.01).bfloat16()
        b0=torch.randn(n)*.1;b1=torch.randn(67)*.01
        x=x0.to('hpu');w=w0.to('hpu');v=w1.to('hpu');b=b0.to('hpu');c=b1.to('hpu');ws=torch.ones(n).to('hpu');sync()
        # Validate fake output metadata without device execution.
        xm=torch.empty((m,k),device='meta',dtype=torch.bfloat16)
        wm=torch.empty((n,k),device='meta',dtype=w0.dtype)
        if fp8:
            aq,sa=torch.ops.gaudi_kernels._fp8_quant(xm)
            assert torch.ops.gaudi_kernels._fp8_mm_f32(aq,wm).shape==(m,n)
        else:assert torch.ops.gaudi_kernels._bf16_mm(xm,wm).shape==(m,n)
        stream=torch.hpu.Stream();graph=torch.hpu.HPUGraph()
        with torch.hpu.graph(graph,stream=stream):
            first=linear_fp8_w8a8_prepared(x,w,ws,b) if fp8 else linear_bf16(x,w,b)
            result=linear_bf16(first,v,c)
        sync();graph.replay(asynchronous=asynchronous);sync()
        expected0=reference(x0,w0,b0,fp8);expected1=reference(expected0,w1,b1,False)
        actual0=first.cpu();actual1=result.cpu()
        e0=float((actual0.double()-expected0.double()).norm()/expected0.double().norm())
        e1=float((actual1.double()-expected1.double()).norm()/expected1.double().norm())
        record={'family':'fp8_w8a8' if fp8 else 'bf16','M':m,'N':n,'K':k,
                'first_rel_l2_to_rounded_contract':e0,'chain_rel_l2_to_rounded_contract':e1,
                'all_outputs':m*n+m*67,'outputs_finite':bool(torch.isfinite(actual0).all() and torch.isfinite(actual1).all())}
        print(json.dumps(record),flush=True)
        assert record['outputs_finite'] and e0<.006 and e1<.009,record
        for _ in range(5):graph.replay(asynchronous=asynchronous)
        sync();event=[];wall=[]
        if count:
            before=count()
            for _ in range(10):graph.replay(asynchronous=asynchronous)
            sync();record['synapse_launches_for_10_replays']=count()-before
            assert record['synapse_launches_for_10_replays']==10,record
        for _ in range(5):
            begin,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
            t=time.perf_counter();begin.record(stream)
            for _ in range(40):graph.replay(asynchronous=asynchronous)
            # CPU replay worker must drain before recording the ending event.
            # This deliberately includes the one-per-batch drain cost.
            if asynchronous:sync()
            end.record(stream);end.synchronize();sync()
            event.append(begin.elapsed_time(end)*1000/40);wall.append((time.perf_counter()-t)*1e6/40)
        pointer=result.data_ptr()
        if fp8:
            qa,sa=torch.ops.gaudi_kernels._fp8_quant(x)
            prod=torch.ops.gaudi_kernels._fp8_mm_f32(qa,w);sync()
            qc=qa.cpu();sc=sa.cpu();pc=prod.cpu()
            exact_div=x0.double()/sc.double()*2
            f32_div=x0.float()/(sc*.5)
            q64=(exact_div.clamp(-448,448).to(torch.float8_e4m3fn).float()*.5).to(torch.float8_e4m3fn)
            q32=(f32_div.clamp(-448,448).to(torch.float8_e4m3fn).float()*.5).to(torch.float8_e4m3fn)
            dot=qc.double()@w0.double().T
            record['quant_diagnostics']={'different_codes_vs_same_scale_f32_div':int((qc.view(torch.uint8)!=q32.view(torch.uint8)).sum()),
                'different_codes_vs_same_scale_f64_div':int((qc.view(torch.uint8)!=q64.view(torch.uint8)).sum()),
                'mme_rel_l2_vs_f64_actual_quantized_operands':float((pc.double()-dot).norm()/dot.norm())}
            assert record['quant_diagnostics']['different_codes_vs_same_scale_f32_div']==0
            assert record['quant_diagnostics']['mme_rel_l2_vs_f64_actual_quantized_operands']<5e-6
            saved[f'quant-{m}']={'q_actual':qc,'scale':sc,'q_f32':q32,'q_f64':q64,'mme_f32':pc,'mme_f64':dot,'input':x0}
        changed=x0.neg();x.copy_(changed);sync();graph.replay(asynchronous=asynchronous);sync()
        changed_ref=reference(reference(changed,w0,b0,fp8),w1,b1,False)
        actual_changed=result.cpu()
        changed_error=float((actual_changed.double()-changed_ref.double()).norm()/changed_ref.double().norm())
        assert changed_error<.009 and pointer==result.data_ptr() and not torch.equal(actual_changed,actual1)
        record.update(event_us=event,wall_us=wall,median_event_us=statistics.median(event),
                      median_wall_us=statistics.median(wall),changed_input_rel_l2=changed_error,
                      stable_output_address=True,asynchronous=asynchronous,status='PASS')
        record['timing_valid']=timings_agree(event,wall)
        assert record['timing_valid'],record
        records.append(record);print(json.dumps(record),flush=True)
        saved[f'{record["family"]}-{m}']={'first':actual0,'chain':actual1,'expected_first':expected0,'expected_chain':expected1,'changed':actual_changed}
torch.save(saved,out/'outputs.pt')
(out/'result.json').write_text(json.dumps({'status':'PASS','records':records,'binding_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
    'scope':'two linears in one HPU Graph replay; not vLLM/model/70-layer quality qualification',
    'underlying_launch_count':'counted' if count else 'not_measured','weight_expansion':'none'},indent=2)+'\n')
