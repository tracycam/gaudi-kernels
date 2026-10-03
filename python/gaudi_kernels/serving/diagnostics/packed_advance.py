# SPDX-License-Identifier: Apache-2.0
"""Mutable-address-stable target replay, including candidate-prefix commit.

Target-only recorder qualification. Preparation and comparison are synchronous
diagnostics; no sampler, drafter, scheduler allocator, explicit graph or TPS.
"""
import ctypes
import json
import time


def on_worker(worker):
    import torch
    import torch.distributed as dist
    import habana_frameworks.torch.core as htcore
    from vllm.distributed import get_tp_group
    from gaudi_kernels.engine.context import context
    from gaudi_kernels.engine.token_batch import RequestTokens, TokenBatch
    from gaudi_kernels.serving.executor.packed_bindings import BoundKVSession, PackedInputBuffers
    from gaudi_kernels.serving.executor.packed_target import PackedTargetExecutor
    from gaudi_kernels.serving.executor.production_quality import compare_logits

    runner = worker.model_runner
    if runner.input_batch.num_reqs or context().native.active:
        raise ValueError('Advancing target recording requires no active requests/native plan')
    root = context().startup.run_dir/'packed-advancing-replay'
    root.mkdir(exist_ok=True)
    api = ctypes.CDLL(None)
    api.e1_begin.argtypes = [ctypes.c_int]
    api.e1_end.argtypes = [ctypes.c_int, ctypes.c_char_p]
    api.e1_replay.argtypes = [ctypes.c_uint, *[ctypes.POINTER(ctypes.c_uint64)]*3]
    api.e1_submission_count.restype = ctypes.c_uint32
    api.e1_set_semantic_role.argtypes = [ctypes.c_char_p]
    holder = ctypes.CDLL(context().path('tensor_hold', 'host'))
    holder.e1_tensor_hold_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p]
    target = PackedTargetExecutor(runner, kv_capacity=128)
    target.session = BoundKVSession(target.session, (1, 4, 8), 32)
    reference = PackedTargetExecutor(runner, kv_capacity=128)
    report = {'rank': worker.rank, 'pass': False, 'steps': [], 'source_captures': 1,
              'scope': 'advancing target recorder with exclusive KV; no sampler, complete MTP or TPS',
              'explicit_graph_producer': False, 'history_capacity_per_request': 32}

    def batch_at(step):
        starts = tuple(len(target.session.arena.committed.get(str(i), ())) for i in range(3))
        return TokenBatch(tuple(RequestTokens(str(i), tuple(101 + i*19 + j*7 + step*11 for j in range(n)),
                                starts[i], 'decode' if n == 1 else 'verify', tuple(range(n)))
                                for i, n in enumerate((1, 4, 8))))

    def vote(ready):
        decision = torch.tensor([int(ready)], dtype=torch.int32)
        dist.all_reduce(decision, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
        return bool(decision.item())

    def snapshot(result):
        return {name: getattr(result, name).detach().cpu().clone() for name in ('hidden', 'logits')}

    def live_snapshot():
        slots = tuple(s for values in target.session.arena.committed.values() for s in values)
        indices = torch.tensor(slots, device=runner.device, dtype=torch.int64)
        return {name: tuple(t.index_select(0, indices).cpu().clone() for t in pair)
                for name, pair in target.session.caches.items()}, indices

    try:
        initial = batch_at(0)
        inputs = PackedInputBuffers(initial, runner.device)
        warm = target.execute_prepared(initial, target.session.prepare(initial), inputs)
        target.abort(warm)
        metadata = target.session.prepare(initial)
        htcore.mark_step()
        torch.hpu.synchronize()
        api.e1_set_semantic_role(None)
        if not vote(api.e1_begin(torch.hpu.current_device()) == 0):
            raise RuntimeError('All-rank advancing recording begin failed')
        try:
            captured = target.execute_prepared(initial, metadata, inputs)
        finally:
            api.e1_pause()
        code = api.e1_end(torch.hpu.current_device(), str(root/f'rank{worker.rank}-commands.txt').encode())
        held = (ctypes.c_uint64*3)()
        holder.e1_tensor_hold_stats(held, str(root/f'rank{worker.rank}-storage.txt').encode())
        report.update(record_code=code, sdk_submission_count=api.e1_submission_count(),
                      storage_lease_counts=list(held))
        if not vote(code == 0 and held[0] > 0 and held[2] > 0):
            raise RuntimeError('Advancing recording did not retain all-rank storage')
        pointers = tuple(t.data_ptr() for t in (inputs.ids, inputs.positions, metadata.slot_mapping,
            *metadata.visible_slots, *metadata.query_positions, *metadata.visible_lengths))
        for step, commits in enumerate(((1, 2, 8), (1, 0, 2), (1, 4, 8), (1, 1, 1))):
            batch = initial if step == 0 else batch_at(step)
            before, live_indices = live_snapshot()
            if step:
                metadata = target.session.prepare(batch)
                inputs.update(batch)
                htcore.mark_step()
                torch.hpu.synchronize()
                times = [ctypes.c_uint64() for _ in range(3)]
                started = time.perf_counter_ns()
                code = api.e1_replay(1, *[ctypes.byref(value) for value in times])
                wall = time.perf_counter_ns() - started
            else:
                code, wall, times = 0, 0, [ctypes.c_uint64() for _ in range(3)]
            actual = snapshot(captured)
            unchanged = all(torch.equal(old.contiguous().view(torch.uint8),
                                new.index_select(0, live_indices).cpu().contiguous().view(torch.uint8))
                            for name, old_pair in before.items()
                            for old, new in zip(old_pair, target.session.caches[name]))
            expected_result = reference.execute(batch)
            expected = snapshot(expected_result)
            checks = [compare_logits(expected['logits'][i:i+1], actual['logits'][i:i+1],
                                     contract='fp32_arithmetic_v1') for i in range(batch.num_tokens)]
            delta = (actual['hidden'].float() - expected['hidden'].float()).norm()
            current = tuple(t.data_ptr() for t in (inputs.ids, inputs.positions, metadata.slot_mapping,
                *metadata.visible_slots, *metadata.query_positions, *metadata.visible_lengths))
            row = {'step': step, 'input_ids': list(batch.encoded()['token_ids']),
                   'starts': [r.start_position for r in batch.requests], 'commits': commits,
                   'replay_code': code, 'replay_wall_ns': wall, 'enqueue_host_ns': times[0].value,
                   'addresses_stable': current == pointers, 'past_kv_bytes_unchanged': unchanged,
                   'checks': checks, 'hidden_relative_l2': (delta/expected['hidden'].float().norm()).item()}
            row['pass'] = code == 0 and unchanged and current == pointers and all(c['pass'] for c in checks)
            report['steps'].append(row)
            torch.save({'actual': actual, 'reference': expected}, root/f'rank{worker.rank}-step{step}.pt')
            if not vote(row['pass']):
                raise RuntimeError('All-rank advancing target output/live-KV check failed')
            target.session.commit(metadata, commits)
            reference.commit(expected_result, commits)
            if target.session.arena.committed != reference.session.committed:
                raise RuntimeError('Captured/reference accepted prefix slot maps differ')
        report['pass'] = True
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
