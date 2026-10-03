"""Same-address packed command replay diagnostic, not an explicit graph producer."""
import ctypes
import json
import time


def on_worker(worker):
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_tp_group
    from gaudi_kernels.engine.context import context
    from gaudi_kernels.engine.token_batch import RequestTokens, TokenBatch
    from gaudi_kernels.serving.executor.packed_target import PackedTargetExecutor

    runner = worker.model_runner
    if runner.input_batch.num_reqs or context().native.active:
        raise ValueError('Packed recording requires an idle scheduler and no native serving plan')
    root = context().startup.run_dir/'packed-static-replay'
    root.mkdir(exist_ok=True)
    api = ctypes.CDLL(None)
    api.e1_begin.argtypes = [ctypes.c_int]
    api.e1_end.argtypes = [ctypes.c_int, ctypes.c_char_p]
    api.e1_replay.argtypes = [ctypes.c_uint, *[ctypes.POINTER(ctypes.c_uint64)]*3]
    api.e1_submission_count.restype = ctypes.c_uint32
    api.e1_set_semantic_role.argtypes = [ctypes.c_char_p]
    holder = ctypes.CDLL(context().path('tensor_hold', 'host'))
    holder.e1_tensor_hold_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p]
    report = {'rank': worker.rank, 'pass': False,
              'scope': 'static same-address target-only recorded replay; no binding updates, MTP, sampling or TPS',
              'explicit_graph_producer': False, 'source_model_calls_per_capture': 1}
    batch = TokenBatch((RequestTokens('d', (101,), 0, 'decode'),
                        RequestTokens('v', (102, 103, 104, 105), 0, 'verify'),
                        RequestTokens('p', tuple(range(111, 119)), 0, 'prefill', tuple(range(8)))))
    executor = PackedTargetExecutor(runner, kv_capacity=32)
    warm = executor.execute(batch)
    executor.abort(warm)

    def vote(ready):
        decision = torch.tensor([int(ready)], dtype=torch.int32)
        dist.all_reduce(decision, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
        return bool(decision.item())

    try:
        api.e1_set_semantic_role(None)
        begin = api.e1_begin(torch.hpu.current_device())
        if not vote(begin == 0):
            raise RuntimeError('All-rank packed recording begin failed')
        try:
            result = executor.execute(batch)
        finally:
            api.e1_pause()
        ended = api.e1_end(torch.hpu.current_device(), str(root/f'rank{worker.rank}-commands.txt').encode())
        held = (ctypes.c_uint64*3)()
        holder.e1_tensor_hold_stats(held, str(root/f'rank{worker.rank}-storage.txt').encode())
        report.update(record_code=ended, storage_lease_counts=list(held),
                      sdk_submission_count=api.e1_submission_count())
        ready = vote(ended == 0 and held[0] > 0 and held[2] > 0)
        if ready:
            before = {'hidden': result.hidden.cpu(), 'logits': result.logits.cpu()}
            report['replays'] = []
            for repeat in range(3):
                times = [ctypes.c_uint64() for _ in range(3)]
                started = time.perf_counter_ns()
                code = api.e1_replay(1, *[ctypes.byref(value) for value in times])
                elapsed = time.perf_counter_ns() - started
                checks = {name: torch.equal(value.contiguous().view(torch.uint8),
                          getattr(result, name).cpu().contiguous().view(torch.uint8)) for name, value in before.items()}
                report['replays'].append({'repeat': repeat, 'code': code, 'checks': checks,
                                         'replay_wall_ns': elapsed,
                                         'host_ns': times[0].value, 'launch_ns': times[1].value,
                                         'hccl_ns': times[2].value})
                if not vote(code == 0 and all(checks.values())):
                    raise RuntimeError('All-rank packed static replay numerical check failed')
            report['pass'] = True
            if worker.rank == 0:
                torch.save(before, root/'reference.pt')
        executor.abort(result)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        api.e1_pause()
        report['release_code'] = holder.native_tensor_hold_release()
        if report['release_code']:
            report['pass'] = False
        (root/f'rank{worker.rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
