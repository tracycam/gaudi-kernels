"""Validate standard foreach H2D copies and graph consumers on one owned HPU."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--module', type=int, default=5)
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = str(args.module)
    import torch
    import habana_frameworks.torch as ht
    from habana_frameworks.torch.activity_profiler.activity_profiler import DebugActivity

    def equal(actual, expected):
        ht.core.mark_step()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    records = []
    with torch.inference_mode():
        for width in (1,64,512):
            sources = [torch.zeros(width,dtype=d) for d in
                       (torch.int32,torch.bfloat16,torch.float32,torch.int64,torch.bfloat16)]
            targets = [x.to('hpu') for x in sources]
            ht.core.mark_step(); ht.hpu.synchronize()
            graph = ht.hpu.HPUGraph()
            with ht.hpu.graph(graph):
                output = sum(x.float() for x in targets)
            graph.replay(); ht.hpu.synchronize()
            for step in range(32):
                for i,x in enumerate(sources):
                    x.fill_((step+i)%13-6)
                expected = sum(x.float() for x in sources)
                torch._foreach_copy_(targets,sources,non_blocking=True)
                # Pageable sources must have been snapshotted by return.
                for x in sources:
                    x.fill_(999)
                graph.replay()
                equal(output,expected)
            arms = []
            for batched in (False,True,True,False):
                for x in sources: x.fill_(2)
                begin = time.perf_counter_ns()
                for _ in range(64):
                    if batched:
                        torch._foreach_copy_(targets,sources,non_blocking=True)
                    else:
                        for dst,src in zip(targets,sources): dst.copy_(src,non_blocking=True)
                    graph.replay()
                    ht.hpu.synchronize()
                arms.append(dict(batched=batched,complete_us=(time.perf_counter_ns()-begin)/64000))
                equal(output,torch.full((width,),10.))
            records.append(dict(width=width,checks=32,timing=arms))
            # This scope proves the batched backend actually ran. It is after
            # timing, and is not a model performance result.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                    debug_activities=[DebugActivity.BRIDGE_FUNCTION_CALLS]) as profile:
                torch._foreach_copy_(targets,sources,non_blocking=True)
                ht.hpu.synchronize()
            trace = args.out/f'copy-trace-{width}.json'
            profile.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())['traceEvents']
            calls = sum(e.get('name')=='BatchedHostToDeviceCopy' and
                        e.get('ph') in ('B','X') for e in events)
            if calls != 1:
                raise AssertionError(f'Batched copy backend coverage: {calls}, expected 1')
            records[-1]['batched_backend_calls'] = calls
            graph.reset()

        # An outstanding D2H write must finish before the H2D staging snapshot.
        gpu = torch.full((257,),37,dtype=torch.int32,device='hpu')
        host = torch.zeros(257,dtype=torch.int32)
        dest = torch.zeros(257,dtype=torch.int32,device='hpu')
        other_host = torch.full((8,),19,dtype=torch.int32)
        other_dest = other_host.to('hpu')
        ht.core.mark_step(); ht.hpu.synchronize()
        dest.copy_(host,non_blocking=True)
        host.copy_(gpu,non_blocking=True)
        torch._foreach_copy_([dest,other_dest],[host,other_host],non_blocking=True)
        equal(dest,torch.full_like(host,37))
        equal(other_dest,other_host)
        records.append(dict(kind='pending-d2h-source',pass_=True))

        # Overlapping destinations, views, casts and int64 retain ordinary
        # sequential copy semantics through the upstream slow implementation.
        dest = torch.zeros(8,device='hpu')
        torch._foreach_copy_([dest,dest],[torch.ones(8),torch.full((8,),3.)],non_blocking=True)
        equal(dest,torch.full((8,),3.))
        view = dest[::2]
        torch._foreach_copy_([view],[torch.full((4,),7.)],non_blocking=True)
        equal(dest,torch.tensor([7.,3.,7.,3.,7.,3.,7.,3.]))
        for dtype in (torch.int64,torch.bfloat16):
            src = torch.arange(12,dtype=torch.float32).reshape(3,4).T
            dst = torch.empty((4,3),dtype=dtype,device='hpu')
            torch._foreach_copy_([dst],[src],non_blocking=True)
            equal(dst,src.to(dtype))
        records.append(dict(kind='fallback-alias-view-cast-layout',pass_=True))
        ht.hpu.synchronize()
    outcomes = []
    for device in ('cpu','hpu'):
        source = torch.arange(8,dtype=torch.float32,requires_grad=True)
        target = torch.empty(8,device=device)
        torch._foreach_copy_([target],[source],non_blocking=True)
        equal(target.detach(),source.detach())
        error = None
        try:
            target.sum().backward()
        except RuntimeError as exc:
            error = str(exc)
        if error is None:
            torch.testing.assert_close(source.grad,torch.ones_like(source),rtol=0,atol=0)
        outcomes.append(error)
    if outcomes[0] != outcomes[1]:
        raise AssertionError(f'Autograd differs from the CPU operator: {outcomes}')
    records.append(dict(kind='autograd-fallback',backward_error=outcomes[0],pass_=True))
    (args.out/'result.json').write_text(json.dumps(dict(status='PASS',records=records,
        scope='Copy/graph microbenchmark only; not model TPS'),indent=2)+'\n')
    print('H2D foreach copy checks PASS',flush=True)


if __name__ == '__main__':
    main()
