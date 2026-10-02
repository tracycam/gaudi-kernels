"""Batch SWA: real operator, independent FP32 oracle, old M1 rows, graph mutation."""
import json,os,time,statistics
from pathlib import Path
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2];out=Path(os.environ['PROBE_OUT'])
torch.ops.load_library(str(root/'build-torch/gaudi_swa128_batch_torch.so'))
torch.ops.load_library(str(root/'libraries/swa_torch.so'))
def sync():hc.mark_step();torch.hpu.synchronize()
def cpu_reference(q,k,v,pages,groups,positions,sinks):
    rows=q.shape[0];result=torch.zeros((rows,16,128))
    for row in range(rows):
        pos=int(positions[row]);selected=(groups==row).nonzero().flatten();slots=[]
        logical0=max(0,pos-127)//128*128
        for n,index in enumerate(selected.tolist()):
            page=int(pages[index]);start=logical0+128*n
            if not 0<=page<k.shape[0]//128:continue
            slots.extend(page*128+t for t in range(max(0,pos-127-start),min(128,pos-start+1)))
        if not slots:continue
        scores=q[row].reshape(16,192).float()@k[slots,0].float().T*(192**-.5)
        probability=torch.cat((scores,sinks.float()[:,None]),-1).softmax(-1)[:,:len(slots)]
        result[row]=probability@v[slots,0].float()
    return result.bfloat16()


records=[];timings=[];torch.manual_seed(87)
with torch.inference_mode():
    kcpu=(torch.randn(20*128,1,192)*.5).bfloat16();vcpu=(torch.randn(20*128,1,128)*.5).bfloat16();scpu=torch.randn(16).bfloat16()
    k,v,s=[t.to('hpu')for t in (kcpu,vcpu,scpu)];sync()
    for rows in (1,4,8,12,24,32):
        qcpu=(torch.randn(rows,1,3072)*.5).bfloat16();q=qcpu.to('hpu')
        pages=torch.zeros(2*rows+3,dtype=torch.int32,device='hpu');groups=torch.full_like(pages,-1)
        positions=torch.zeros((rows,1),dtype=torch.int32,device='hpu');sync();stream=torch.hpu.Stream();graphs={};outputs={}
        for name in ('window_fp32','window_quad_fp32'):
            graph=torch.hpu.HPUGraph()
            with torch.hpu.graph(graph,stream=stream):outputs[name]=getattr(torch.ops.gaudi_swa128_batch,name)(q+0,k,v,pages+0,groups+0,positions+0,s,192**-.5)
            graphs[name]=graph
        sync()
        for shift in (0,1,127,128):
            pp=torch.zeros(2*rows+3,dtype=torch.int32);gg=torch.full_like(pp,-1)
            pos=torch.tensor([(0,126,127,128,129,255,4095,4096,32767)[i%9]+shift for i in range(rows)],dtype=torch.int32)
            cursor=0
            for i,n in enumerate(pos.tolist()):
                if rows>1 and i==rows-1:continue  # ragged padding, no valid pages
                first=max(0,n-127)//128;last=n//128
                for logical in range(first,last+1):pp[cursor]=(3*i+7*logical)%20;gg[cursor]=i;cursor+=1
            pages.copy_(pp);groups.copy_(gg);positions.copy_(pos[:,None]);sync()
            for graph in graphs.values():graph.replay()
            sync();actual=outputs['window_quad_fp32'].cpu();scalar=outputs['window_fp32'].cpu()
            assert torch.equal(actual,scalar),dict(rows=rows,shift=shift,variant='quad_vs_scalar')
            expected=cpu_reference(qcpu,kcpu,vcpu,pp,gg,pos,scpu)
            torch.testing.assert_close(actual,expected,rtol=.01,atol=.0002)
            m1=[]
            for i in range(rows):
                indices=(gg==i).nonzero().flatten();p=pp[indices]
                if not len(p):p=torch.zeros(1,dtype=torch.int32);g=torch.full_like(p,-1)
                else:g=torch.zeros_like(p)
                qi=qcpu[i:i+1].contiguous().to('hpu');pi=p.to('hpu');gi=g.to('hpu');po=pos[i:i+1].to('hpu')
                yy=torch.ops.gaudi_swa128.forward_window_fast_fp32(qi,k,v,pi,gi,po,s,192**-.5);sync();m1.append(yy.cpu())
            old=torch.cat(m1,0);bitwise=torch.equal(actual,old)
            row=dict(rows=rows,shift=shift,old_m1_bitwise=bitwise,quad_scalar_bitwise=True,max_abs_fp32_reference=float((actual.float()-expected.float()).abs().max()))
            records.append(row);assert bitwise,row
        # Same stream, same process ABBA. Includes launch/host supply; not engine-only.
        for graph in graphs.values():
            for _ in range(3):graph.replay(asynchronous=True)
        sync();paired=[]
        for repeat in range(3):
            for name in ('window_fp32','window_quad_fp32','window_quad_fp32','window_fp32'):
                graph=graphs[name];start,end=torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
                begin=time.perf_counter()
                with torch.hpu.stream(stream):
                    start.record(stream)
                    for _ in range(100):graph.replay(asynchronous=True)
                    end.record(stream)
                end.synchronize();sync()
                paired.append(dict(repeat=repeat,variant=name,event_us=start.elapsed_time(end)*10,wall_us=(time.perf_counter()-begin)*1e4))
        medians={name:statistics.median(v['event_us']for v in paired if v['variant']==name)for name in graphs}
        timings.append(dict(rows=rows,paired=paired,median_event_us=medians,speedup=medians['window_fp32']/medians['window_quad_fp32']))
        del graph,graphs,outputs;sync()
        (out/'result.json').write_text(json.dumps(dict(status='RUNNING',records=records,timings=timings),indent=2)+'\n')
(out/'result.json').write_text(json.dumps(dict(status='PASS_BATCH_SWA_OPERATOR',records=records,timings=timings,
    scope='One batched kernel R1..32; causal per-query pages, independent FP32 oracle, exact old-M1 operator comparison. Paired scalar/quad same-stream launch timing. Not serving-qualified.'),indent=2)+'\n')
