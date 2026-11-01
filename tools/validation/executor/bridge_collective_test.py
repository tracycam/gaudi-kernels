"""Independent Torch compute/collective/consumer graph; no model or private replay API."""
import argparse
import json
import os
from pathlib import Path
import socket
import time
import subprocess
import sys
import faulthandler


def worker(rank, modules, address, output, require_native, timing_replays, cpu_trace=False,
           region_chains=False, require_coalesced=False, event_batching=False, compiled_regions=False,
           fault_prefix=False, defrag_cache=False, gil_progress=False):
    os.environ['HLS_MODULE_ID'] = str(modules[rank])
    faulthandler.enable()
    print(f'COLLECTIVE_PROBE rank={rank} module={modules[rank]} importing', flush=True)
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
    print(f'COLLECTIVE_PROBE rank={rank} initialized HCCL', flush=True)
    control = dist.new_group(backend='gloo')
    secondary = dist.new_group(backend='hccl') if region_chains else None
    records = []
    graph = None
    try:
        for dtype in (torch.float32, torch.bfloat16):
            kinds = ['all-reduce', 'all-gather', 'mixed-rank-fallback', 'collective-only']
            if region_chains:
                kinds += ['chain-8', 'chain-32', 'independent-consumer', 'two-communicators']
                if event_batching or compiled_regions:
                    kinds += ['mixed-chain-8']
            for kind in kinds:
                chain = 8 if kind == 'mixed-chain-8' else int(kind.split('-')[1]) if kind.startswith('chain-') else 0
                mixed = kind in ('mixed-rank-fallback', 'mixed-chain-8')
                rebound = mixed and rank == 1
                print(f'COLLECTIVE_PROBE rank={rank} dtype={dtype} kind={kind} capture', flush=True)
                host = torch.full((8, 8), rank+1., dtype=dtype)
                x = host.to('hpu')
                weight = torch.ones((8, 8), dtype=dtype, device='hpu')
                ht.core.mark_step()
                ht.hpu.synchronize()
                graph = ht.hpu.HPUGraph()
                with torch.inference_mode():
                    graph.capture_begin()
                    if rebound:
                        graph.mark_user_inputs([x])
                    if chain:
                        value = x
                        for _ in range(chain):
                            value = (value + rank + 1) * .125
                            ht.core.mark_step()
                            dist.all_reduce(value)
                            ht.core.mark_step()
                        y = value * 3 + 1
                    elif kind == 'collective-only':
                        # No compute recipe, including at graph exit. Completion
                        # must remain visible to an ordinary external consumer.
                        dist.all_reduce(x)
                        ht.core.mark_step()
                        gathered = torch.empty((16, 8), dtype=dtype, device='hpu')
                        dist.all_gather_into_tensor(gathered, x)
                        ht.core.mark_step()
                        y = gathered
                    else:
                        value = x @ weight
                        ht.core.mark_step()
                    if kind == 'all-gather':
                        gathered = torch.empty((16, 8), dtype=dtype, device='hpu')
                        dist.all_gather_into_tensor(gathered, value)
                    elif kind != 'collective-only' and not chain:
                        dist.all_reduce(value)
                        gathered = value
                    if kind == 'two-communicators':
                        ht.core.mark_step()
                        gathered = value * .125
                        ht.core.mark_step()
                        dist.all_reduce(gathered,group=secondary)
                    if kind != 'collective-only' and not chain:
                        ht.core.mark_step()
                        y = (x if kind == 'independent-consumer' else gathered) * 3 + 1
                    graph.capture_end()
                print(f'COLLECTIVE_PROBE rank={rank} captured {kind}', flush=True)
                ht.hpu.synchronize()
                before = (_hpu_C.native_replay_stats(graph.hpu_graph)
                          if hasattr(_hpu_C, 'native_replay_stats') else None)
                if (event_batching or compiled_regions) and before and before['ready']:
                    setter = _hpu_C.native_replay_compiled_regions if compiled_regions else _hpu_C.native_replay_event_batching
                    setter(graph.hpu_graph, True)
                participants = [None]*len(modules)
                # Diagnostic agreement outside the captured graph and timing.
                # This does not pretend the runtime has added a consensus API.
                dist.all_gather_object(participants, before, group=control)
                if require_native:
                    for peer, state in enumerate(participants):
                        admitted = not mixed or peer == 0
                        if not state or state['ready'] != admitted:
                            raise AssertionError(f'capture admission differs: {kind}: {participants}')
                        if admitted and state['collectives'] != (chain or (2 if kind in ('collective-only','two-communicators') else 1)):
                            raise AssertionError(f'collective missing from plan: {participants}')
                        if require_coalesced and admitted:
                            expected_mapped = chain
                            if state.get('coalesced_collectives') != expected_mapped:
                                raise AssertionError(f'Incorrect regional stream admission: {kind}: {participants}')
                with torch.inference_mode():
                    for index in range(8):
                        x.copy_((host+index).to('hpu'))
                        ht.core.mark_step()
                        print(f'COLLECTIVE_PROBE rank={rank} kind={kind} replay={index}', flush=True)
                        if rebound:
                            graph.replayV3([x])
                        else:
                            graph.replay()
                        actual = y.cpu()
                        if chain:
                            # Match legal FP32/BF16 operation rounding at each
                            # stage, independently of the device graph runtime.
                            values = [torch.full((8,8),peer+1+index,dtype=dtype)
                                      for peer in range(len(modules))]
                            for _ in range(chain):
                                values = [(v + peer + 1) * .125 for peer,v in enumerate(values)]
                                total = values[0] + values[1]
                                values = [total,total]
                            expected = total * 3 + 1
                        elif kind == 'independent-consumer':
                            expected = (host+index)*3+1
                        elif kind == 'collective-only':
                            expected = torch.full((16,8), 3+2*index, dtype=dtype)
                        elif kind == 'all-gather':
                            expected = torch.cat([torch.full((8,8), 8*(peer+1+index), dtype=dtype)
                                                  for peer in range(len(modules))])
                        else:
                            expected = torch.full((8,8), 8*(3+2*index), dtype=dtype)
                        if kind == 'two-communicators':
                            expected = expected * .125 * 2
                        if kind != 'collective-only' and not chain and kind != 'independent-consumer':
                            expected = expected * 3 + 1
                        if not torch.equal(actual, expected):
                            torch.save(dict(actual=actual, expected=expected),
                                       Path(output)/f'failure-rank{rank}-{dtype}-{kind}-{index}.pt')
                            raise AssertionError(f'{dtype}, {kind}, rank {rank}, input {index}')
                    ht.hpu.synchronize()
                    dist.barrier(group=control)
                    start = time.perf_counter_ns()
                    for _ in range(timing_replays):
                        if rebound:
                            graph.replayV3([x])
                        else:
                            graph.replay()
                    submitted = time.perf_counter_ns()
                    ht.hpu.synchronize()
                    completed = time.perf_counter_ns()
                after = (_hpu_C.native_replay_stats(graph.hpu_graph)
                         if hasattr(_hpu_C, 'native_replay_stats') else None)
                if require_native and not rebound:
                    if after['replays'] != 8+timing_replays or after['queued_replays'] != after['replays']:
                        raise AssertionError(f'mixed replay fell back or queued more than once: {after}')
                if event_batching and chain and not rebound and (not after.get('event_batches') or
                        after.get('interrupted_event_batches')):
                    raise AssertionError(f'Event batch admission/coverage failed: {after}')
                if compiled_regions and chain and not rebound and not after.get('prepared_region_replays'):
                    raise AssertionError(f'Prepared region was not exercised: {after}')
                records.append(dict(dtype=str(dtype), kind=kind, checks=8, before=before, after=after,
                                    timing_replays=timing_replays,
                                    enqueue_us=(submitted-start)/1e3/timing_replays,
                                    complete_us=(completed-start)/1e3/timing_replays))
                if region_chains and chain and not mixed:
                    # Both ordinary HCCL and a graph-external stream surround
                    # replay. These calls deliberately have no intermediate
                    # device synchronization and use the same communicator.
                    marker = torch.full((8,8),rank+10.,dtype=dtype,device='hpu')
                    ht.core.mark_step()
                    dist.all_reduce(marker)
                    ht.core.mark_step()
                    graph.replay()
                    dist.all_reduce(marker)
                    ht.core.mark_step()
                    consumer_stream = ht.hpu.Stream()
                    with ht.hpu.stream(consumer_stream):
                        outside = y + 2
                        ht.core.mark_step()
                    if not torch.equal(marker.cpu(),torch.full((8,8),42.,dtype=dtype)):
                        raise AssertionError('Graph mapping violated ordinary communicator FIFO')
                    if not torch.equal(outside.cpu(),expected+2):
                        raise AssertionError('Graph completion missing for external stream consumer')
                    records[-1]['ordinary_collective_fifo_and_external_consumer'] = True
                    if event_batching or compiled_regions:
                        graph.replay()
                        with ht.hpu.stream(consumer_stream):
                            x.fill_(123)
                            ht.core.mark_step()
                        if not torch.equal(y.cpu(), expected):
                            raise AssertionError('External writer overtook in-flight graph reads')
                        records[-1]['external_writer_waits_for_graph_reads'] = True
                if cpu_trace:
                    from habana_frameworks.torch.activity_profiler.activity_profiler import DebugActivity
                    # Trace after the uninstrumented timing. Both ranks issue
                    # exactly the same additional collectives, even in mixed mode.
                    with torch.inference_mode(), torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU],
                            debug_activities=[DebugActivity.BRIDGE_FUNCTION_CALLS]) as profile:
                        for _ in range(8):
                            with torch.profiler.record_function('HPUGraph.replay'):
                                if rebound:
                                    graph.replayV3([x])
                                else:
                                    graph.replay()
                        ht.hpu.synchronize()
                    trace = Path(output)/f'cpu-trace-rank{rank}-{dtype}-{kind}.json'
                    profile.export_chrome_trace(str(trace))
                    records[-1]['cpu_trace'] = str(trace)
                    records[-1]['after_trace'] = _hpu_C.native_replay_stats(graph.hpu_graph)
                if kind == 'collective-only' or (region_chains and kind == 'chain-8'):
                    if kind == 'collective-only':
                        with torch.inference_mode():
                            x.copy_((host+31).to('hpu'))
                            ht.core.mark_step()
                            graph.replay()
                            graph.reset()
                            graph = None
                            actual = y.cpu()
                        if not torch.equal(actual, torch.full((16,8),65.,dtype=dtype)):
                            raise AssertionError('Collective output became invalid after in-flight reset')
                        records[-1]['reset_before_external_consumer'] = True
                    else:
                        graph.reset()
                        graph = None

                    # Hold the peer off the next replay: rank0's collectives
                    # cannot finish before the Gloo release barrier. Unlike the
                    # consumer check above, no Python Tensor owns this output
                    # after reset. A weak Storage handle observes actual lease
                    # ownership without keeping the allocation alive itself.
                    lease_input = host.to('hpu')
                    lease_output = torch.empty((16,8),dtype=dtype,device='hpu')
                    ht.core.mark_step()
                    lease_graph = ht.hpu.HPUGraph()
                    with torch.inference_mode():
                        lease_graph.capture_begin()
                        lease_value = lease_input + 1 if kind == 'chain-8' else lease_input
                        if kind == 'chain-8':
                            ht.core.mark_step()
                        dist.all_reduce(lease_value)
                        ht.core.mark_step()
                        gather_input = lease_value * 2 if kind == 'chain-8' else lease_value
                        if kind == 'chain-8':
                            ht.core.mark_step()
                        dist.all_gather_into_tensor(lease_output,gather_input)
                        ht.core.mark_step()
                        if kind == 'chain-8':
                            lease_output.add_(1)
                        lease_graph.capture_end()
                    ht.hpu.synchronize()
                    state = _hpu_C.native_replay_stats(lease_graph.hpu_graph)
                    if (event_batching or compiled_regions) and state['ready']:
                        setter = _hpu_C.native_replay_compiled_regions if compiled_regions else _hpu_C.native_replay_event_batching
                        setter(lease_graph.hpu_graph, True)
                        state = _hpu_C.native_replay_stats(lease_graph.hpu_graph)
                    if require_native and (not state['ready'] or state['collectives'] != 2):
                        raise AssertionError('Lease test did not capture native collectives')
                    if require_coalesced and kind == 'chain-8' and state.get('coalesced_collectives') != 2:
                        raise AssertionError('Peer-blocked lease test did not map the dependent region')
                    records[-1]['peer_blocked_lease_plan'] = state
                    weak = lease_output.untyped_storage()._weak_ref() if rank == 0 else None
                    dist.barrier(group=control)
                    alive = None
                    if rank == 0:
                        lease_graph.replay()
                        records[-1]['peer_blocked_submission'] = _hpu_C.native_replay_stats(lease_graph.hpu_graph)
                        if compiled_regions and kind == 'chain-8' and not records[-1]['peer_blocked_submission']['prepared_region_replays']:
                            raise AssertionError('Blocked peer test did not submit the prepared region')
                        lease_graph.reset()
                        lease_graph = None
                        del lease_output
                        alive = not torch.UntypedStorage._expired(weak)
                    # Rank1 deliberately has not submitted either collective
                    # until rank0 has observed the reset graph's output lease.
                    dist.barrier(group=control)
                    if rank == 1:
                        lease_graph.replay()
                    ht.hpu.synchronize()
                    observations = [None]*len(modules)
                    dist.all_gather_object(observations,alive,group=control)
                    if weak is not None:
                        records[-1]['output_storage_expired_after_completion'] = torch.UntypedStorage._expired(weak)
                        torch.UntypedStorage._free_weak_ref(weak)
                    if lease_graph is not None:
                        lease_graph.reset()
                    if observations[0] is not True:
                        raise AssertionError('Output Storage destroyed while peer-blocked collective is in flight')
                    records[-1]['output_storage_alive_during_peer_blocked_reset'] = True
                else:
                    graph.reset()
                    graph = None
        if defrag_cache:
            # Force fragmentation through ordinary Tensor allocations, never a
            # device reset or an artificial epoch increment. No host-sized copy.
            import gc
            blocks = []
            large = None
            with torch.inference_mode():
                ht.hpu.synchronize()
                memory = torch.hpu.memory_stats()
                available = memory['Limit'] - memory['InUse']
                chunk_rows = int(available * .90 / 24) // 4096
                large_rows = int(available * .18) // 4096
                if chunk_rows < 1024:
                    raise AssertionError('Insufficient idle memory for defrag qualification')
                try:
                    block = torch.empty((chunk_rows,1024),dtype=torch.float32,device='hpu')
                    block.fill_(0); ht.core.mark_step(); ht.hpu.synchronize()
                    blocks.append(block); del block
                    x = torch.full((8,8),float(rank+1),device='hpu')
                    ht.core.mark_step()
                    graph = ht.hpu.HPUGraph()
                    graph.capture_begin()
                    a = x + 1; ht.core.mark_step()
                    dist.all_reduce(a); ht.core.mark_step()
                    b = a * 2; ht.core.mark_step()
                    dist.all_reduce(b); ht.core.mark_step()
                    y = b + 3
                    graph.capture_end()
                    _hpu_C.native_replay_compiled_regions(graph.hpu_graph,True)
                    graph.replay(); graph.replay(); ht.hpu.synchronize()
                    before = _hpu_C.native_replay_stats(graph.hpu_graph)
                    if not before.get('address_cache_hits'):
                        raise AssertionError('No retained address-cache hit before memory pressure')
                    dist.barrier(group=control)
                    for _ in range(23):
                        block = torch.empty((chunk_rows,1024),dtype=torch.float32,device='hpu')
                        block.fill_(0); ht.core.mark_step(); ht.hpu.synchronize()
                        blocks.append(block); del block
                    for index in range(0,len(blocks),2):
                        blocks[index] = None
                    gc.collect(); ht.hpu.synchronize()
                    fragmented = torch.hpu.memory_stats()
                    large = torch.empty((large_rows,1024),dtype=torch.float32,device='hpu')
                    large.fill_(1); ht.core.mark_step(); ht.hpu.synchronize()
                    dist.barrier(group=control)
                    graph.replay()
                    actual = y.cpu()
                    after = _hpu_C.native_replay_stats(graph.hpu_graph)
                    if not torch.equal(actual,torch.full((8,8),23.)):
                        raise AssertionError('Replay after allocator relocation produced wrong values')
                    if after['address_cache_misses'] <= before['address_cache_misses']:
                        raise AssertionError('Memory pressure did not exercise relocation invalidation')
                    records.append(dict(kind='allocator-relocation', before=before, after=after,
                        initial_memory=memory, fragmented_memory=fragmented,
                        chunk_bytes=chunk_rows*4096, large_bytes=large_rows*4096,
                        relocation_invalidated_cache=True, output_equal=True))
                finally:
                    ht.hpu.synchronize()
                    if graph is not None: graph.reset(); graph = None
                    large = None; blocks.clear(); gc.collect(); ht.hpu.synchronize()
        if gil_progress:
            import ctypes
            import threading
            api = ctypes.CDLL(None)
            api.launch_pause_arm.argtypes = [ctypes.c_uint]
            api.launch_pause_wait_entered.argtypes = [ctypes.c_uint]
            api.launch_pause_wait_entered.restype = ctypes.c_int
            api.launch_pause_expired.restype = ctypes.c_int
            for prepared in (False, True):
                with torch.inference_mode():
                    x = torch.full((8,8),float(rank+1),device='hpu')
                    ht.core.mark_step()
                    graph = ht.hpu.HPUGraph()
                    graph.capture_begin()
                    a = x+1; ht.core.mark_step()
                    dist.all_reduce(a); ht.core.mark_step()
                    b = a*2; ht.core.mark_step()
                    dist.all_reduce(b); ht.core.mark_step()
                    y = b+3
                    graph.capture_end(); ht.hpu.synchronize()
                    _hpu_C.native_replay_compiled_regions(graph.hpu_graph, prepared)
                    dist.barrier(group=control)
                    progress, errors = [], []
                    def release_from_python():
                        try:
                            # ctypes releases the GIL while waiting here. It
                            # must reacquire it before the Python release call.
                            if not api.launch_pause_wait_entered(5000):
                                raise AssertionError('Submission never reached pause')
                            progress.append(True)
                            api.launch_pause_release()
                            # These may race replay's return; their graph-lock
                            # waits must not hold the GIL either.
                            _hpu_C.native_replay_stats(graph.hpu_graph)
                            graph.reset()
                        except BaseException as error:
                            errors.append(str(error))
                    api.launch_pause_arm(2)
                    helper = threading.Thread(target=release_from_python,daemon=True)
                    helper.start()
                    try:
                        graph.replay()
                    finally:
                        api.launch_pause_release()
                        helper.join(timeout=5)
                    if helper.is_alive() or errors or not progress or api.launch_pause_expired():
                        raise AssertionError(f'Graph submission blocked Python progress: {errors}')
                    api.launch_pause_arm(0)
                    ht.hpu.synchronize()
                    if not torch.equal(y.cpu(),torch.full((8,8),23.)):
                        raise AssertionError('Concurrent graph reset changed output')
                    graph = None
                    records.append(dict(kind='gil-progress', prepared=prepared,
                        python_callback_before_submission_return=True,
                        concurrent_snapshot_and_reset=True, output_equal=True))
        if fault_prefix:
            import ctypes
            api = ctypes.CDLL(None)
            api.launch_fault_arm.argtypes = [ctypes.c_uint]
            api.launch_fault_calls.restype = ctypes.c_uint
            with torch.inference_mode():
                x = torch.ones((8,8), device='hpu')
                ht.core.mark_step()
                graph = ht.hpu.HPUGraph()
                graph.capture_begin()
                z = x * 2
                ht.core.mark_step()
                dist.all_reduce(z)
                ht.core.mark_step()
                mid = z + 1
                ht.core.mark_step()
                dist.all_reduce(mid)
                ht.core.mark_step()
                y = mid * 3
                graph.capture_end()
                ht.hpu.synchronize()
                _hpu_C.native_replay_compiled_regions(graph.hpu_graph, True)
                before = _hpu_C.native_replay_stats(graph.hpu_graph)
                if not before['ready'] or not before['prepared_regions']:
                    raise AssertionError('Fault test lacks a prepared region')
                dist.barrier(group=control)
                # Both ranks submit the same first collective, then fail the
                # following recipe. No peer is left waiting for an absent rank.
                api.launch_fault_arm(2)
                try:
                    try:
                        graph.replay()
                    except RuntimeError as error:
                        if 'Prepared recipe launch failed' not in str(error):
                            raise AssertionError('Failure did not occur inside the prepared region') from error
                    else:
                        raise AssertionError('Injected failure was not raised')
                    failed = _hpu_C.native_replay_stats(graph.hpu_graph)
                    if not failed['failed'] or failed['ready'] or api.launch_fault_calls() != 2:
                        raise AssertionError(f'Failed prefix was not poisoned: {failed}')
                    try:
                        graph.replay()
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError('Poisoned graph accepted another replay')
                    if api.launch_fault_calls() != 2:
                        raise AssertionError('Poisoned replay resubmitted an SDK command')
                finally:
                    api.launch_fault_arm(0)
                graph.reset()
                graph = None
                ht.hpu.synchronize()
                records.append(dict(kind='failed-prepared-prefix', before=before, after=failed,
                                    error_propagated=True, retry_submissions=0, reset_and_drain=True))
        libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                            if 'libhabana_pytorch' in line or 'libe1_' in line or 'libgkg' in line})
        if any('libe1_' in path or 'libgkg' in path for path in libraries):
            raise AssertionError('External executor loaded')
        from types import SimpleNamespace
        from gaudi_kernels.serving.host_placement import snapshot
        placement_index = placement['workers'].index(local)
        physical_placement = snapshot(SimpleNamespace(rank=rank, local_rank=placement_index))
        result = dict(status='PASS', rank=rank, module=modules[rank], records=records,
                      libraries=libraries, cpu_affinity=sorted(os.sched_getaffinity(0)),
                      physical_placement=physical_placement,
                      scope='Two-rank normal graph functional/hot-replay probe; not model TPS')
        (Path(output)/f'rank{rank}.json').write_text(json.dumps(result, indent=2)+'\n')
    finally:
        if graph is not None:
            ht.hpu.synchronize()
            graph.reset()
        dist.destroy_process_group(control)
        if secondary is not None:
            dist.destroy_process_group(secondary)
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--modules', nargs=2, type=int, default=(5, 4))
    parser.add_argument('--require-native', action='store_true')
    parser.add_argument('--timing-replays', type=int, default=64)
    parser.add_argument('--worker-rank', type=int, choices=(0,1))
    parser.add_argument('--address')
    parser.add_argument('--gdb-rank', type=int, choices=(0,1))
    parser.add_argument('--cpu-trace', action='store_true',
                        help='CPU bridge spans after timing; no hardware profiler')
    parser.add_argument('--region-chains', action='store_true',
                        help='Also test 8/32 dependent collectives, independent branch and communicator FIFO')
    parser.add_argument('--require-coalesced', action='store_true',
                        help='Require eligible regions to use the mapped stream and independent branches to keep theirs')
    parser.add_argument('--event-batching', action='store_true')
    parser.add_argument('--compiled-regions', action='store_true')
    parser.add_argument('--fault-prefix', action='store_true')
    parser.add_argument('--defrag-cache', action='store_true')
    parser.add_argument('--gil-progress', action='store_true')
    args = parser.parse_args()
    if len(set(args.modules)) != 2 or any(module not in range(8) for module in args.modules):
        parser.error('two distinct module IDs required')
    if args.timing_replays < 1:
        parser.error('positive replay count required')
    if args.worker_rank is not None:
        worker(args.worker_rank, tuple(args.modules), args.address, str(args.out),
               args.require_native, args.timing_replays, args.cpu_trace,
               args.region_chains, args.require_coalesced, args.event_batching, args.compiled_regions,
               args.fault_prefix, args.defrag_cache, args.gil_progress)
        return
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        address = f'tcp://127.0.0.1:{sock.getsockname()[1]}'
    children = []
    placement = json.loads((args.out/'host-placement.json').read_text())
    try:
        for rank, module in enumerate(args.modules):
            env = dict(os.environ, HLS_MODULE_ID=str(module), RANK=str(rank),
                       LOCAL_RANK=str(rank), WORLD_SIZE='2',
                       HABANA_LOGS=str(args.out/f'habana-rank{rank}'))
            # Select the module before Torch's backend autoload, matching the
            # existing serving worker factory. Children inherit the supervisor
            # process group so its bounded cleanup also owns both ranks.
            command = [sys.executable, '-m', __spec__.name, '--out', str(args.out),
                       '--modules', *map(str, args.modules), '--worker-rank', str(rank),
                       '--address', address, '--timing-replays', str(args.timing_replays)]
            if args.require_native:
                command.append('--require-native')
            if args.cpu_trace:
                command.append('--cpu-trace')
            if args.region_chains:
                command.append('--region-chains')
            if args.require_coalesced:
                command.append('--require-coalesced')
            if args.event_batching:
                command.append('--event-batching')
            if args.compiled_regions:
                command.append('--compiled-regions')
            if args.fault_prefix:
                command.append('--fault-prefix')
            if args.defrag_cache:
                command.append('--defrag-cache')
            if args.gil_progress:
                command.append('--gil-progress')
            if placement['binding'] == 'local':
                local = next(item for item in placement['workers'] if item['module_id'] == module)
                command = ['numactl','--physcpubind='+','.join(map(str,local['allowed_local_cpus'])),
                           '--membind='+str(local['numa_node']),*command]
            if args.gdb_rank == rank:
                command = ['gdb', '--batch', '-ex', 'set pagination off',
                           '-ex', 'set confirm off', '-ex', 'run', '-ex', 'bt 30',
                           '-ex', 'quit 88', '--args', *command]
            children.append(subprocess.Popen(command, env=env))
        while any(child.poll() is None for child in children):
            failed = [child.returncode for child in children if child.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'owned rank failed: {failed}')
            time.sleep(.1)
        if any(child.returncode for child in children):
            raise RuntimeError('owned rank failed')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
    ranks = [json.loads((args.out/f'rank{rank}.json').read_text()) for rank in range(2)]
    (args.out/'result.json').write_text(json.dumps(dict(status='PASS', ranks=ranks), indent=2)+'\n')
    print('Two-rank compute/collective/consumer checks PASS', flush=True)


if __name__ == '__main__':
    main()
