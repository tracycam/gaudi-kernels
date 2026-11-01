"""Model-free, bounded TP8 all-gather completion and numeric checks.

Uses ordinary PyTorch and its public HPUGraph API. Per-call completion includes
host submission, so this is a health/tail-latency diagnostic, not raw RoCE BW.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def worker(rank, modules, out, iterations, graph_mode, vendor, interval_ms, pair_layout, operation):
    os.environ['HLS_MODULE_ID'] = str(modules[rank])
    os.environ['GC_KERNEL_PATH'] = vendor
    placement = json.loads((Path(out)/'host-placement.json').read_text())
    local = next(w for w in placement['workers'] if w['module_id'] == modules[rank])
    if placement['binding'] == 'local':
        os.sched_setaffinity(0, local['allowed_local_cpus'])
    import torch
    import torch.distributed as dist
    import habana_frameworks.torch as ht
    import habana_frameworks.torch.distributed.hccl
    torch.set_num_threads(4)
    dist.init_process_group('hccl', init_method='file://'+str(Path(out)/'rendezvous'),
                            rank=rank, world_size=len(modules))
    graph = None
    try:
        group = None
        peers = tuple(range(len(modules)))
        transport = None
        if pair_layout == 'hierarchical':
            from gaudi_kernels.serving.distributed.rank_gather import GatherLayout, RankGatherTransport
            transport = RankGatherTransport(dist.group.WORLD, GatherLayout(
                ((0,1,2,3),(4,5,6,7)), ((0,5),(1,6),(2,7),(3,4))))
        elif pair_layout != 'world':
            pairs = ((0,4),(1,5),(2,6),(3,7)) if pair_layout == 'cross-half' else ((0,1),(2,3),(4,5),(6,7))
            for pair in pairs:
                candidate = dist.new_group(ranks=list(pair), backend='hccl')
                if rank in pair:
                    group, peers = candidate, pair
        x = torch.full((1,6144), rank+1., dtype=torch.float32, device='hpu')
        y = torch.empty((len(peers) if operation == 'all_gather' else 1,6144), dtype=torch.float32, device='hpu')
        ht.core.mark_step(); ht.hpu.synchronize()
        expected = torch.tensor([p+1. for p in peers],dtype=torch.float32)[:,None].expand(len(peers),6144)
        if operation == 'all_reduce':
            expected = expected.sum(0,keepdim=True)
        def collective():
            if operation == 'all_reduce':
                y.copy_(x)
                if transport:
                    transport.all_reduce_float32(y)
                else:
                    dist.all_reduce(y,group=group)
                return
            if transport:
                transport.all_gather_into_tensor(y,x)
            else:
                dist.all_gather_into_tensor(y,x,group=group)
        if graph_mode:
            graph = ht.hpu.HPUGraph()
            graph.capture_begin()
            collective()
            ht.core.mark_step()
            graph.capture_end()
        def execute():
            if graph_mode:
                graph.replay()
            else:
                collective(); ht.core.mark_step()
            ht.hpu.synchronize()
        with torch.inference_mode():
            for _ in range(8):
                execute()
            torch.testing.assert_close(y.cpu(),expected,rtol=0,atol=0)
            for offset in (16.,32.,0.):
                x.fill_(rank+1.+offset)
                ht.core.mark_step(); ht.hpu.synchronize()
                execute()
                torch.testing.assert_close(y.cpu(),expected+offset*(len(peers) if operation == 'all_reduce' else 1),rtol=0,atol=0)
            samples = []
            starts_utc_ns = []
            for _ in range(iterations):
                starts_utc_ns.append(time.time_ns())
                begin = time.perf_counter_ns(); execute()
                samples.append((time.perf_counter_ns()-begin)/1000)
                if interval_ms:
                    time.sleep(interval_ms/1000)
            torch.testing.assert_close(y.cpu(),expected,rtol=0,atol=0)
        mapped = sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines()
                         if any(n in l for n in ('libSynapse.so','libhcl.so','libscal.so','libhl-thunk.so'))})
        result = dict(status='PASS_NUMERIC',rank=rank,module=modules[rank],iterations=iterations,
                      graph=graph_mode,complete_us=samples,p50_us=statistics.median(samples),
                      pair_layout=pair_layout, peers=peers,
                      operation=operation,
                      starts_utc_ns=starts_utc_ns,
                      max_us=max(samples),above_100ms=sum(v>100000 for v in samples),
                      communication_environment={name: os.environ.get(name) for name in (
                          'HCL_FAULT_TOLERANCE_ENABLE', 'LOGICAL_SCALE_OUT_PORTS_MASK',
                          'LOG_LEVEL_HCL', 'LOG_LEVEL_HCL_IBV', 'LOG_LEVEL_HCL_FAILOVER')},
                      loaded_runtime_libraries=mapped,
                      execution=graph.execution_info() if graph and hasattr(graph,'execution_info') else None)
        (Path(out)/f'rank-{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        if graph:
            ht.hpu.synchronize(); graph.reset()
        dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=256)
    p.add_argument('--graph',action='store_true')
    p.add_argument('--operation',choices=('all_gather','all_reduce'),default='all_gather')
    p.add_argument('--pair-layout',choices=('world','adjacent','cross-half','hierarchical'),default='world',
                   help='Equal-size pairs diagnose peer-path dependence; rank numbering, not arbitrary port masking')
    p.add_argument('--interval-ms',type=float,default=5.)
    p.add_argument('--vendor-tpc',type=Path,required=True)
    a=p.parse_args()
    if not 0<=a.interval_ms<=20:
        p.error('interval must be bounded to 0..20 ms')
    if not 32<=a.iterations<=2048:
        p.error('iterations must be bounded to 32..2048')
    modules=tuple(int(v) for v in os.environ['HABANA_VISIBLE_MODULES'].split(','))
    if len(modules)!=8:
        p.error('TP8 is required')
    import torch.multiprocessing as mp
    mp.spawn(worker,args=(modules,str(a.out.resolve()),a.iterations,a.graph,str(a.vendor_tpc.resolve(strict=True)),a.interval_ms,a.pair_layout,a.operation),
             nprocs=len(modules),join=True)
    records=[json.loads((a.out/f'rank-{i}.json').read_text()) for i in range(len(modules))]
    (a.out/'result.json').write_text(json.dumps(dict(status='PASS_NUMERIC',ranks=records,
            scope='TP8 all-gather completion diagnostic; no model, not raw network bandwidth'),indent=2)+'\n')


if __name__=='__main__':
    main()
