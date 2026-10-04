"""Independent Torch compute/collective/consumer graph; no model or private replay API."""
import argparse
import json
import os
from pathlib import Path
import socket
import time


def worker(rank, modules, address, output, require_native, timing_replays):
    os.environ['HABANA_VISIBLE_MODULES'] = str(modules[rank])
    os.environ['HLS_MODULE_ID'] = str(modules[rank])
    placement = json.loads((Path(output)/'host-placement.json').read_text())
    local = next(item for item in placement['workers'] if item['module_id'] == modules[rank])
    if placement['binding'] == 'local':
        for task in Path('/proc/self/task').iterdir():
            try:
                os.sched_setaffinity(int(task.name), local['allowed_local_cpus'])
            except ProcessLookupError:
                pass
    import torch
    import torch.distributed as dist
    import habana_frameworks.torch as ht
    import habana_frameworks.torch.distributed.hccl  # register backend
    from habana_frameworks.torch import _hpu_C
    dist.init_process_group('hccl', rank=rank, world_size=len(modules), init_method=address)
    control = dist.new_group(backend='gloo')
    records = []
    graph = None
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for kind in ('all-reduce', 'all-gather', 'mixed-rank-fallback'):
                host = torch.full((8, 8), rank+1., dtype=dtype)
                x = host.to('hpu')
                weight = torch.ones((8, 8), dtype=dtype, device='hpu')
                ht.core.mark_step()
                ht.hpu.synchronize()
                graph = ht.hpu.HPUGraph()
                with torch.inference_mode():
                    graph.capture_begin()
                    if kind == 'mixed-rank-fallback' and rank == 1:
                        graph.mark_user_inputs([x])
                    value = x @ weight
                    ht.core.mark_step()
                    if kind == 'all-gather':
                        gathered = torch.empty((16, 8), dtype=dtype, device='hpu')
                        dist.all_gather_into_tensor(gathered, value)
                    else:
                        dist.all_reduce(value)
                        gathered = value
                    ht.core.mark_step()
                    y = gathered * 3 + 1
                    graph.capture_end()
                ht.hpu.synchronize()
                before = (_hpu_C.native_replay_stats(graph.hpu_graph)
                          if hasattr(_hpu_C, 'native_replay_stats') else None)
                participants = [None]*len(modules)
                # Diagnostic agreement outside the captured graph and timing.
                # This does not pretend the runtime has added a consensus API.
                dist.all_gather_object(participants, before, group=control)
                if require_native:
                    for peer, state in enumerate(participants):
                        admitted = kind != 'mixed-rank-fallback' or peer == 0
                        if not state or state['ready'] != admitted:
                            raise AssertionError(f'capture admission differs: {kind}: {participants}')
                        if admitted and state['collectives'] != 1:
                            raise AssertionError(f'collective missing from plan: {participants}')
                with torch.inference_mode():
                    for index in range(8):
                        x.copy_((host+index).to('hpu'))
                        ht.core.mark_step()
                        graph.replay()
                        actual = y.cpu()
                        if kind == 'all-gather':
                            expected = torch.cat([torch.full((8,8), 8*(peer+1+index), dtype=dtype)
                                                  for peer in range(len(modules))])
                        else:
                            expected = torch.full((8,8), 8*(3+2*index), dtype=dtype)
                        expected = expected * 3 + 1
                        if not torch.equal(actual, expected):
                            torch.save(dict(actual=actual, expected=expected),
                                       Path(output)/f'failure-rank{rank}-{dtype}-{kind}-{index}.pt')
                            raise AssertionError(f'{dtype}, {kind}, rank {rank}, input {index}')
                    ht.hpu.synchronize()
                    dist.barrier(group=control)
                    start = time.perf_counter_ns()
                    for _ in range(timing_replays):
                        graph.replay()
                    submitted = time.perf_counter_ns()
                    ht.hpu.synchronize()
                    completed = time.perf_counter_ns()
                after = (_hpu_C.native_replay_stats(graph.hpu_graph)
                         if hasattr(_hpu_C, 'native_replay_stats') else None)
                if require_native and (kind != 'mixed-rank-fallback' or rank == 0):
                    if after['replays'] != 8+timing_replays or after['queued_replays'] != after['replays']:
                        raise AssertionError(f'mixed replay fell back or queued more than once: {after}')
                records.append(dict(dtype=str(dtype), kind=kind, checks=8, before=before, after=after,
                                    timing_replays=timing_replays,
                                    enqueue_us=(submitted-start)/1e3/timing_replays,
                                    complete_us=(completed-start)/1e3/timing_replays))
                graph.reset()
                graph = None
        libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                            if 'libhabana_pytorch' in line or 'libe1_' in line or 'libgkg' in line})
        if any('libe1_' in path or 'libgkg' in path for path in libraries):
            raise AssertionError('External executor loaded')
        result = dict(status='PASS', rank=rank, module=modules[rank], records=records,
                      libraries=libraries, cpu_affinity=sorted(os.sched_getaffinity(0)),
                      scope='Two-rank normal graph functional/hot-replay probe; not model TPS')
        (Path(output)/f'rank{rank}.json').write_text(json.dumps(result, indent=2)+'\n')
    finally:
        if graph is not None:
            ht.hpu.synchronize()
            graph.reset()
        dist.destroy_process_group(control)
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--modules', nargs=2, type=int, default=(5, 4))
    parser.add_argument('--require-native', action='store_true')
    parser.add_argument('--timing-replays', type=int, default=64)
    args = parser.parse_args()
    if len(set(args.modules)) != 2 or any(module not in range(8) for module in args.modules):
        parser.error('two distinct module IDs required')
    if args.timing_replays < 1:
        parser.error('positive replay count required')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        address = f'tcp://127.0.0.1:{sock.getsockname()[1]}'
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(tuple(args.modules), address, str(args.out), args.require_native,
                          args.timing_replays), nprocs=2, join=True)
    ranks = [json.loads((args.out/f'rank{rank}.json').read_text()) for rank in range(2)]
    (args.out/'result.json').write_text(json.dumps(dict(status='PASS', ranks=ranks), indent=2)+'\n')
    print('Two-rank compute/collective/consumer checks PASS', flush=True)


if __name__ == '__main__':
    main()
