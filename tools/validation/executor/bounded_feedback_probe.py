"""Bounded device feedback through the existing bridge; synthetic, not LLM TPS.

Each step uses BF16 MME, FP32 all-gather/sum, greedy sampling, and device
state/KV updates. The Python loop builds a static graph; replay has no host
token feedback. SDK calls are counted separately from application calls.
"""
import argparse
import ctypes
import fcntl
import json
import mmap
import os
from pathlib import Path
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def worker(a):
    # Environment and NUMA affinity are selected before loading Torch.
    rank = a.rank
    import torch
    import torch.distributed as dist
    import habana_frameworks.torch as ht
    import habana_frameworks.torch.distributed.hccl
    from habana_frameworks.torch import _hpu_C
    torch.set_num_threads(1)
    world = len(a.modules)
    dist.init_process_group('hccl', rank=rank, world_size=world, init_method=a.address)
    control = dist.new_group(backend='gloo')
    api = ctypes.CDLL(None)
    api.api_timing_end.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    torch.ops.load_library(str(a.gate_library))
    graphs = []
    records = []
    try:
        with torch.inference_mode():
            width = 64
            vocab = torch.arange(width, device='hpu')
            # Complementary TP contributions: alternating output columns live
            # on different ranks. A local-only or duplicated-peer gather cannot
            # pass the recurrence oracle. Their sum is the cyclic transition.
            weight_host = torch.roll(torch.eye(width, dtype=torch.bfloat16), 1, 1)
            weight_host *= (torch.arange(width) % world == rank).to(torch.bfloat16)
            weight = weight_host.to('hpu')
            x = position = active = count = eos = kv = history = cursor = None
            states = {}

            def select(steps):
                nonlocal x, position, active, count, eos, kv, history, cursor
                x, position, active, count, eos, kv, history, cursor = states[steps]

            def native_execution(graph):
                info = _hpu_C.native_replay_stats(graph.hpu_graph)
                if not info['ready'] or info['failed'] or not info['replays']:
                    raise AssertionError(('native replay fell back', info))
                if os.environ['PT_HPU_ENABLE_SFG'] == '0' and info['collectives'] > 1:
                    if not info['prepared_commands'] or not info['prepared_region_replays']:
                        raise AssertionError(('prepared replay not exercised', info))
                return info

            def reset(seed=2, stop=-1):
                host_x = torch.zeros((1, width), dtype=torch.bfloat16)
                host_x[0, seed] = 1
                x.copy_(host_x.to('hpu'))
                # Host-origin copies preserve runtime input storage; lazy
                # fill/zero origins can otherwise become capture constants.
                position.copy_(torch.tensor([126],dtype=torch.int64).to('hpu'))
                active.copy_(torch.tensor([True],dtype=torch.bool).to('hpu'))
                count.copy_(torch.tensor([0],dtype=torch.int64).to('hpu'))
                eos.copy_(torch.tensor([stop],dtype=torch.int64).to('hpu'))
                kv.copy_(torch.full((128,),-1,dtype=torch.int64).to('hpu'))
                if a.repeat_plan:
                    cursor.copy_(torch.tensor([0],dtype=torch.int64).to('hpu'))
                    history.copy_(torch.full((history.numel(),),-1,dtype=torch.int32).to('hpu'))
                ht.core.mark_step(); ht.hpu.synchronize()

            def step():
                value = x
                for depth in range(a.collectives_per_step):
                    local = (value @ weight).float()
                    ht.core.mark_step()
                    gathered = torch.empty((world, width), dtype=torch.float32, device='hpu')
                    dist.all_gather_into_tensor(gathered, local)
                    ht.core.mark_step()
                    if depth+1 < a.collectives_per_step:
                        value = gathered.sum(0).reshape(1, width).to(torch.bfloat16)
                token = gathered.sum(0).argmax().reshape(1)
                emitted = torch.where(active, token, -torch.ones_like(token))
                slot = position.remainder(128)
                previous = kv.gather(0, slot)
                kv.copy_(kv.scatter(0, slot, torch.where(active, token, previous)))
                x.copy_(torch.where(active.reshape(1, 1),
                                    (vocab == token).to(torch.bfloat16).reshape(1, width), x))
                # Explicit stores back to persistent input buffers. Lazy
                # in-place add/scatter alone may create a captured output
                # storage distinct from the next replay's input storage.
                position.copy_(position + active.to(torch.int64))
                count.copy_(count + active.to(torch.int64))
                active.copy_(active & (token != eos))
                ht.core.mark_step()
                return emitted

            def build(steps):
                # Each independently captured program owns mutable state.
                # Capturing another graph must not rewrite its Lazy bindings.
                states[steps] = (torch.zeros((1,width),dtype=torch.bfloat16,device='hpu'),
                    torch.zeros(1,dtype=torch.int64,device='hpu'),
                    torch.ones(1,dtype=torch.bool,device='hpu'),
                    torch.zeros(1,dtype=torch.int64,device='hpu'),
                    torch.full((1,),-1,dtype=torch.int64,device='hpu'),
                    torch.full((128,),-1,dtype=torch.int64,device='hpu'),
                    torch.full((steps,),-1,dtype=torch.int32,device='hpu') if a.repeat_plan else None,
                    torch.zeros(1,dtype=torch.int64,device='hpu') if a.repeat_plan else None)
                select(steps)
                reset()
                # Warm operators outside capture, then restore state.
                step(); ht.core.mark_step(); ht.hpu.synchronize(); reset()
                graph = ht.hpu.HPUGraph()
                graph.capture_begin()
                if a.repeat_plan:
                    emitted = step().to(torch.int32)
                    history.copy_(history.scatter(0,cursor.remainder(steps),emitted))
                    cursor.copy_(cursor+1)
                    values = None
                else:
                    values = [step() for _ in range(steps)]
                # Explicit physical ABI for the diagnostic DMA witness. HPU
                # logical int64 tensors may use 32-bit device storage.
                result = history if a.repeat_plan else torch.cat(values).to(torch.int32)
                ht.core.mark_step(); graph.capture_end(); ht.hpu.synchronize()
                before = _hpu_C.native_replay_stats(graph.hpu_graph)
                if not before['ready'] or before['collectives'] != (1 if a.repeat_plan else steps)*a.collectives_per_step:
                    raise AssertionError(('native admission', steps, before))
                _hpu_C.native_replay_compiled_regions(graph.hpu_graph, True)
                graphs.append(graph)
                print('CAPTURE', rank, steps, before, flush=True)
                return graph, result

            steps_to_build = (1, 4, 16) if not a.capacity_only else tuple(a.block_steps)
            blocks = {steps: build(steps) for steps in steps_to_build}

            def execute(steps):
                graph = blocks[steps][0]
                if a.repeat_plan: graph.replay_many(steps)
                else: graph.replay()

            def check(steps, seed, stop):
                select(steps)
                reset(seed, stop)
                graph, output = blocks[steps]
                execute(steps); ht.hpu.synchronize()
                native_execution(graph)
                expected = []
                expected_kv = [-1] * 128
                value, pos, running, valid = seed, 126, True, 0
                for _ in range(steps):
                    if running:
                        value = (value + a.collectives_per_step) % width
                        expected.append(value); expected_kv[pos % 128] = value
                        pos += 1; valid += 1; running = value != stop
                    else:
                        expected.append(-1)
                got = dict(tokens=output.cpu().tolist(), position=position.cpu().item(),
                           active=active.cpu().item(), count=count.cpu().item(), kv=kv.cpu().tolist())
                want = dict(tokens=expected, position=pos, active=running, count=valid, kv=expected_kv)
                if got != want:
                    save(a.out / f'failure-rank{rank}.json', dict(got=got, want=want))
                    raise AssertionError(('device feedback', steps, seed, stop))
                return dict(steps=steps, seed=seed, eos=stop, actual=got)

            fixtures = ((2, -1), (61, -1), (2, (2+3*a.collectives_per_step)%width),
                        (2, (2+a.collectives_per_step)%width))
            if a.capacity_only: fixtures = (fixtures[0],fixtures[2])
            checks = [check(n, seed, stop) for n in steps_to_build for seed, stop in fixtures]

            # A correct first block is insufficient: every mutable buffer must
            # feed the next invocation. Also continue once after EOS.
            continuous_checks = []
            for steps in steps_to_build:
                for stop in (-1,(2+3*a.collectives_per_step)%width):
                    select(steps); reset(2,stop)
                    value,pos,running,valid = 2,126,True,0
                    expected_kv = [-1]*128
                    for cycle in range(2):
                        execute(steps); ht.hpu.synchronize()
                        native_execution(blocks[steps][0])
                        tokens = []
                        for _ in range(steps):
                            if running:
                                value = (value+a.collectives_per_step)%width
                                tokens.append(value);expected_kv[pos%128]=value
                                pos+=1;valid+=1;running=value!=stop
                            else: tokens.append(-1)
                        actual = dict(tokens=blocks[steps][1].cpu().tolist(),
                            position=position.cpu().item(),active=active.cpu().item(),
                            count=count.cpu().item(),kv=kv.cpu().tolist())
                        expected = dict(tokens=tokens,position=pos,active=running,count=valid,kv=expected_kv.copy())
                        if actual != expected:
                            save(a.out/f'continuous-failure-rank{rank}.json',dict(
                                steps=steps,cycle=cycle,eos=stop,actual=actual,expected=expected))
                            raise AssertionError('device state did not persist across block replay')
                        continuous_checks.append(dict(steps=steps,cycle=cycle,eos=stop,actual=actual))

            # Separate multiple queued API invocations from an expanded block.
            # Both must preserve repeated in-place state dependencies.
            queued_checks = []
            timed_steps = () if a.capacity_only and not a.measure_capacity else (4,16)
            for steps in timed_steps:
                reference = check(steps, 2, -1)['actual']
                select(1)
                reset()
                for _ in range(steps): blocks[1][0].replay()
                ht.hpu.synchronize()
                actual = dict(position=position.cpu().item(), active=active.cpu().item(),
                              count=count.cpu().item(), kv=kv.cpu().tolist(),
                              final_token=blocks[1][1].cpu().item())
                expected = {k: reference[k] for k in ('position', 'active', 'count', 'kv')}
                expected['final_token'] = reference['tokens'][-1]
                if actual != expected: raise AssertionError(('queued state dependency', actual, expected))
                queued_checks.append(dict(steps=steps, actual=actual))

            # Same operations/state updates; A has per-step device completion
            # and host sampling read, B invokes one expanded block graph.
            for steps in timed_steps:
                for mode in ('per-step-feedback', 'step-queued', 'block',
                             'block', 'step-queued', 'per-step-feedback'):
                    samples = []
                    select(steps if mode=='block' else 1)
                    reset()
                    for _ in range(8):
                        if mode == 'block':
                            execute(steps); blocks[steps][1].cpu()
                        elif mode == 'per-step-feedback':
                            for _ in range(steps):
                                blocks[1][0].replay(); blocks[1][1].cpu()
                        else:
                            for _ in range(steps): blocks[1][0].replay()
                            blocks[1][1].cpu()
                    reset()
                    dist.barrier(group=control)
                    for _ in range(a.rounds):
                        start = time.perf_counter_ns()
                        if mode == 'block':
                            execute(steps)
                            blocks[steps][1].cpu()
                        elif mode == 'per-step-feedback':
                            for _ in range(steps):
                                blocks[1][0].replay()
                                blocks[1][1].cpu()
                        else:
                            for _ in range(steps): blocks[1][0].replay()
                            blocks[1][1].cpu()
                        samples.append((time.perf_counter_ns() - start) / 1e3)
                    records.append(dict(steps=steps, mode=mode, block_us=samples,
                                        median_us_per_step=statistics.median(samples) / steps))
                    print('TIMING', rank, steps, mode, records[-1]['median_us_per_step'], flush=True)

            # Count an isolated replay AFTER uninstrumented timing.
            counts = []
            for steps in steps_to_build:
                select(steps)
                reset(); api.api_timing_begin()
                execute(steps)
                data = (ctypes.c_uint64 * 28)(); api.api_timing_end(data)
                ht.hpu.synchronize()
                counts.append(dict(steps=steps, application_replays=1,
                    apis={name: dict(count=data[4*i], inclusive_ns=data[4*i+1])
                          for i, name in enumerate(('launch', 'launch_ext', 'all_gather',
                              'all_reduce', 'event_record', 'stream_wait', 'event_sync'))},
                    execution=native_execution(blocks[steps][0])))

            save(a.out / f'partial-rank{rank}.json', dict(checks=checks, queued_checks=queued_checks,
                                                       continuous_checks=continuous_checks,
                                                       timings=records, sdk_counts=counts))

            if a.trace_library and not a.capacity_only:
                select(16)
                from habana_frameworks.torch.utils.experimental import _data_ptr
                trace = ctypes.CDLL(str(a.trace_library))
                trace.trace_device.argtypes = [ctypes.c_uint32]
                trace.trace_required.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
                trace.trace_start.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
                trace.trace_stop.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
                if trace.trace_device(torch.hpu.current_device()): raise RuntimeError('trace device')
                required = ctypes.c_uint32()
                if trace.trace_required(ctypes.byref(required)): raise RuntimeError('trace buffer query')
                if required.value > 512*1024*1024: raise ValueError('trace buffer exceeds 512MiB')
                storage = torch.empty(max(4096, required.value), dtype=torch.uint8, device='hpu')
                ht.core.mark_step(); ht.hpu.synchronize(); reset()
                dist.barrier(group=control)
                if trace.trace_start(int(_data_ptr(storage)), storage.numel()): raise RuntimeError('trace start')
                execute(16); ht.hpu.synchronize()
                entries = (ctypes.c_uint64*2)()
                if trace.trace_stop(str(a.out/f'device-rank{rank}.jsonl').encode(), entries):
                    raise RuntimeError('trace stop')

            # A bounded gate provides time to stop ALL threads of both owned
            # workers. The supervisor observes device DMA into mapped host pages.
            # This proves prequeued autonomy only, never a one-SDK-call program.
            pause_steps = max(blocks)
            select(pause_steps)
            reset()
            output = blocks[pause_steps][1]
            from habana_frameworks.torch.utils.experimental import _data_ptr
            source = int(_data_ptr(output))
            # Anonymous private pages are pin-compatible; filesystem-backed
            # mmap pages were rejected by this driver. The parent can read its
            # owned child's mapped page through /proc/PID/mem while stopped.
            with mmap.mmap(-1, 4096, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS) as shared:
                shared[:] = bytes([0xff]) * 4096
                pointer = ctypes.addressof(ctypes.c_char.from_buffer(shared))
                api.feedback_host_map.argtypes = [ctypes.c_uint, ctypes.c_uint64, ctypes.c_void_p]
                api.feedback_copy.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
                api.feedback_host_unmap.argtypes = [ctypes.c_uint, ctypes.c_void_p]
                device = torch.hpu.current_device()
                code = api.feedback_host_map(device, 4096, pointer)
                if code: raise RuntimeError(('host_map', code))
                try:
                    gate_x = torch.ones(64, dtype=torch.float32, device='hpu')
                    gate = ht.hpu.wrap_in_hpu_graph_func(torch.ops.gk_diagnostic_gate.delay_loops)
                    gate(gate_x, a.gate_loops); ht.core.mark_step(); ht.hpu.synchronize()
                    dist.barrier(group=control)
                    held = gate(gate_x, a.gate_loops)
                    code = api.feedback_gate_record(device)
                    if code: raise RuntimeError(('gate event record', code))
                    gate_before = api.feedback_gate_query()
                    dist.all_reduce(held); ht.core.mark_step()
                    # Marker ties gate completion into communicator FIFO,
                    # even if the native region uses another compute stream.
                    if api.api_timing_begin(): raise RuntimeError('API timing overlap')
                    submit_start = time.perf_counter_ns()
                    execute(pause_steps)
                    submit_ns = time.perf_counter_ns()-submit_start
                    counters = (ctypes.c_uint64*28)(); api.api_timing_end(counters)
                    peaks = (ctypes.c_uint64*7)(); api.api_timing_peaks(peaks)
                    peak_indices = (ctypes.c_uint64*7)(); api.api_timing_peak_indices(peak_indices)
                    native_execution(blocks[pause_steps][0])
                    gate_after = api.feedback_gate_query()
                    code = api.feedback_copy(source, pause_steps*4, pointer)
                    if code: raise RuntimeError(('copy', code))
                    snapshot = list(struct.unpack(f'{pause_steps}i', shared[:pause_steps*4]))
                    expected = [(2+(i+1)*a.collectives_per_step)%width for i in range(pause_steps)]
                    save(a.out / f'queued-rank{rank}.json', dict(pid=os.getpid(),
                        initial_dma=snapshot, completion_address=pointer,
                        expected=expected, gate_before=gate_before, gate_after=gate_after,
                        submit_ms=submit_ns/1e6, steps=pause_steps,
                        collectives_per_step=a.collectives_per_step,
                        apis={name:dict(count=counters[4*i], inclusive_ns=counters[4*i+1], peak_ns=peaks[i],
                                        peak_call_index=peak_indices[i])
                              for i,name in enumerate(('launch','launch_ext','all_gather','all_reduce',
                                                       'event_record','stream_wait','event_sync'))}))
                    # Supervisor SIGSTOPs the process while gate is pending.
                    while not (a.out / 'resume.json').exists(): time.sleep(.01)
                    ht.hpu.synchronize()
                    actual = list(struct.unpack(f'{pause_steps}i', shared[:pause_steps*4]))
                    if actual != expected:
                        raise AssertionError(('paused device DMA', actual))
                finally:
                    ht.hpu.synchronize(); api.feedback_gate_destroy(); api.feedback_host_unmap(device, pointer)
            save(a.out / f'rank{rank}.json', dict(status='PASS', rank=rank,
                checks=checks, queued_checks=queued_checks,continuous_checks=continuous_checks,
                timings=records, sdk_counts=counts,
                final_execution={str(n): native_execution(g)
                                 for n, (g, _) in blocks.items()},
                scope='Synthetic MME/collective/device-state probe; not model TPS'))
    finally:
        ht.hpu.synchronize()
        for graph in graphs: graph.reset()
        dist.destroy_process_group()


def supervise(a):
    a.out.mkdir(parents=True, exist_ok=False)
    import hashlib
    source = Path(__file__).resolve().read_bytes()
    (a.out/'source-probe.py').write_bytes(source)
    artifact_paths = [a.gate_library.resolve(strict=True),
                      Path(os.environ['LD_PRELOAD']).resolve(strict=True)]
    if a.trace_library: artifact_paths.append(a.trace_library.resolve(strict=True))
    save(a.out/'source-identity.json', dict(probe_sha256=hashlib.sha256(source).hexdigest(),
        artifacts={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact_paths},
        environment={k: os.environ.get(k) for k in ('PT_HPU_LAZY_MODE','PT_HPU_ENABLE_SFG',
            'PT_HPU_ENABLE_LAZY_COLLECTIVES','PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS',
            'PT_HPU_LAZY_ACC_PAR_MODE','LD_LIBRARY_PATH','GC_KERNEL_PATH')}))
    locks = []
    lock_root = Path.home() / 'gaudi-llm-experiments/.codex-kernel-locks'
    lock_root.mkdir(parents=True, exist_ok=True)
    for module in a.modules:
        handle = (lock_root / f'module-{module}.lock').open('a+')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(handle)
    before = subprocess.check_output(['hl-smi', '-Q', 'module_id,memory.used,utilization.aip',
                                      '-f', 'csv,noheader'], text=True)
    (a.out / 'devices-before.txt').write_text(before)
    for line in before.splitlines():
        module, memory, utilization = [int(v.split()[0]) for v in line.split(',')]
        if module in a.modules and (memory > 1024 or utilization):
            raise RuntimeError(('device busy', module, memory, utilization))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); address = f'tcp://127.0.0.1:{sock.getsockname()[1]}'
    children, stopped, logs = [], [], []
    begin = time.monotonic()
    world = len(a.modules)
    result = dict(status='RUNNING', modules=a.modules, timeout_seconds=a.timeout_seconds,
                  collectives_per_step=a.collectives_per_step,
                  block_steps=a.block_steps, gate_loops=a.gate_loops, capacity_only=a.capacity_only,
                  repeat_plan=a.repeat_plan)
    try:
        for rank, module in enumerate(a.modules):
            environment = dict(os.environ, HLS_MODULE_ID=str(module), RANK=str(rank),
                               LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                               HABANA_LOGS=str(a.out / f'habana-rank{rank}'))
            environment.pop('HABANA_VISIBLE_MODULES', None)
            cpus = '0-27,56-83' if module < 4 else '28-55,84-111'
            command = ['numactl', '--physcpubind='+cpus, '--membind='+str(module//4),
                sys.executable, '-m', __spec__.name, '--rank', str(rank), '--address', address,
                '--out', str(a.out), '--gate-library', str(a.gate_library), '--rounds', str(a.rounds),
                '--modules', *map(str,a.modules), '--collectives-per-step', str(a.collectives_per_step),
                '--gate-loops', str(a.gate_loops), '--block-steps', *map(str,a.block_steps)]
            if a.capacity_only: command.append('--capacity-only')
            if a.measure_capacity: command.append('--measure-capacity')
            if a.repeat_plan: command.append('--repeat-plan')
            if a.trace_library: command += ['--trace-library', str(a.trace_library)]
            log = (a.out / f'worker-rank{rank}.log').open('w'); logs.append(log)
            children.append(subprocess.Popen(command, env=environment, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True))
        while not all((a.out/f'queued-rank{rank}.json').exists() for rank in range(world)):
            if any(child.poll() is not None for child in children):
                raise RuntimeError(('worker exited', [child.poll() for child in children]))
            if time.monotonic() - begin > a.timeout_seconds-45: raise TimeoutError('probe preparation')
            time.sleep(.05)
        queued = [json.loads((a.out/f'queued-rank{rank}.json').read_text()) for rank in range(world)]
        result['queued'] = queued
        if any(q['gate_before'] != 22 or q['gate_after'] not in (0,22) for q in queued):
            raise AssertionError(('gate event not valid', queued))
        limited = any(q['gate_after'] == 0 for q in queued)
        result['prequeue_window_pass'] = not limited
        if limited:
            # A completed gate after CPU replay means the whole chain was not
            # proved prequeued behind it. Preserve numerical completion, but
            # never claim autonomous full-block execution from this sample.
            save(a.out/'resume.json', dict(resume=True, capacity_limited=True))
            for child in children:
                if child.wait(timeout=30): raise RuntimeError(('worker failed',child.returncode))
            result.update(status='FUNCTIONAL_PASS_PREQUEUE_LIMITED',
                          ranks=[json.loads((a.out/f'rank{rank}.json').read_text()) for rank in range(world)])
            return
        if any(q['initial_dma'] != [-1]*len(q['expected']) for q in queued):
            raise AssertionError('Output completed despite pending marker gate')
        for child in children:
            os.kill(child.pid, signal.SIGSTOP); stopped.append(child)
        pause_start = time.perf_counter_ns()
        witness = []
        pending_confirmed = False
        while True:
            witness = []
            for child, q in zip(children, queued):
                state = Path(f'/proc/{child.pid}/status').read_text()
                with Path(f'/proc/{child.pid}/mem').open('rb', buffering=0) as memory:
                    n = len(q['expected'])
                    values = list(struct.unpack(f'{n}i', os.pread(memory.fileno(), n*4,
                                                               q['completion_address'])))
                thread_states = {task.name: next(line for line in (task/'status').read_text().splitlines()
                                                if line.startswith('State:'))
                                 for task in Path(f'/proc/{child.pid}/task').iterdir()}
                witness.append(dict(pid=child.pid, process_stopped='State:\tT' in state,
                                    all_threads_stopped=all('State:\tT' in s for s in thread_states.values()),
                                    thread_count=len(thread_states),
                                    tokens=values, expected=q['expected']))
            if not pending_confirmed and all(w['all_threads_stopped'] for w in witness):
                if any(w['tokens'] != [-1]*len(w['expected']) for w in witness):
                    raise AssertionError(('completion was not pending at confirmed host stop', witness))
                result['stopped_pending_witness'] = witness
                save(a.out / 'stopped-pending-witness.json', result)
                pending_confirmed = True
            if all(w['process_stopped'] and w['all_threads_stopped'] and
                   w['tokens'] == w['expected'] for w in witness): break
            if time.perf_counter_ns() - pause_start > 5_000_000_000:
                raise TimeoutError(('device progress while host stopped', witness))
            time.sleep(.01)
        result['paused_witness'] = witness
        result['pause_to_dma_ms'] = (time.perf_counter_ns() - pause_start)/1e6
        save(a.out / 'paused-witness.json', result)
        save(a.out / 'resume.json', dict(resume=True))
        for child in stopped: os.kill(child.pid, signal.SIGCONT)
        stopped.clear()
        for child in children:
            if child.wait(timeout=30): raise RuntimeError(('worker failed', child.returncode))
        result.update(status='PASS', ranks=[json.loads((a.out/f'rank{rank}.json').read_text())
                                            for rank in range(world)])
    except BaseException as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        for child in stopped:
            if child.poll() is None: os.kill(child.pid, signal.SIGCONT)
        for child in children:
            if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
        for log in logs: log.close()
        result['worker_exit_codes'] = [child.returncode for child in children]
        result['elapsed_seconds'] = time.monotonic() - begin
        save(a.out / 'result.json', result)
        after = subprocess.check_output(['hl-smi', '-Q', 'module_id,memory.used,utilization.aip',
                                         '-f', 'csv,noheader'], text=True)
        (a.out / 'devices-after.txt').write_text(after)


def main():
    # Lazy collectives are required for capture. Production prepared regions
    # use SFG=0; SFG=1 remains an explicit diagnostic control, not the default.
    # Set both before importing the distributed backend in either process.
    os.environ.setdefault('PT_HPU_ENABLE_SFG', '0')
    os.environ.setdefault('PT_HPU_ENABLE_LAZY_COLLECTIVES', 'true')
    os.environ.setdefault('PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS', '1')
    os.environ.setdefault('PT_HPU_LAZY_ACC_PAR_MODE', '0')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gate-library', type=Path, required=True)
    p.add_argument('--trace-library', type=Path)
    p.add_argument('--modules', nargs='+', type=int, default=[0, 1])
    p.add_argument('--rounds', type=int, default=32)
    p.add_argument('--rank', type=int)
    p.add_argument('--address')
    p.add_argument('--capacity-only', action='store_true')
    p.add_argument('--repeat-plan', action='store_true',
                   help='Use bridge replay_many with one captured step and persistent output history')
    p.add_argument('--measure-capacity', action='store_true',
                   help='Time the scaled synthetic chain too; never a model TPS estimate')
    p.add_argument('--collectives-per-step', type=int, default=1)
    p.add_argument('--block-steps', nargs='+', type=int, default=[1,4,16])
    p.add_argument('--gate-loops', type=int, default=50000000)
    p.add_argument('--timeout-seconds', type=int, default=300)
    a = p.parse_args()
    if not 8 <= a.rounds <= 128: p.error('rounds must be 8..128')
    if len(set(a.modules)) != len(a.modules) or not 2<=len(a.modules)<=8 or any(m not in range(8) for m in a.modules):
        p.error('2..8 distinct module IDs required')
    if not 1<=a.collectives_per_step<=142: p.error('collectives per step must be 1..142')
    if not a.capacity_only and a.collectives_per_step!=1: p.error('scaled graph requires capacity-only')
    if not a.block_steps or any(not 1<=n<=16 for n in a.block_steps): p.error('block steps must be 1..16')
    if a.measure_capacity and not {1,4,16}.issubset(a.block_steps): p.error('capacity timing needs 1/4/16 graphs')
    if not 1<=a.gate_loops<=150000000: p.error('bounded gate loops required')
    if not 60<=a.timeout_seconds<=1200: p.error('bounded timeout required')
    a.out = a.out.resolve()
    if a.rank is None: supervise(a)
    else: worker(a)


if __name__ == '__main__': main()
