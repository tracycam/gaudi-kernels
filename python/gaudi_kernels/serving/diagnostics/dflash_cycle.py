# SPDX-License-Identifier: Apache-2.0
"""Actual checkpoint DFlash + target cycle, with private KV and device IDs.

Eager functional smoke test. Precision comparisons and service/native throughput
are separate; generated output may be budget-limited rather than a full answer.
"""
import json
import time


def on_worker(worker, plan):
    import torch
    import torch.distributed as dist
    import habana_frameworks.torch.core as htcore
    from vllm.distributed import get_tp_group
    from gaudi_kernels.engine.context import context
    from gaudi_kernels.engine.token_batch import QueryBatch, RequestQueries, RequestTokens, TokenBatch
    from gaudi_kernels.serving.draft.dflash import load_checkpoint
    from gaudi_kernels.serving.draft.cycle import DFlashCycle
    from gaudi_kernels.serving.executor.packed_bindings import BoundKVSession
    from gaudi_kernels.serving.executor.packed_bindings import PackedInputBuffers
    from gaudi_kernels.serving.executor.packed_target import PackedTargetExecutor

    runner = worker.model_runner
    if runner.input_batch.num_reqs or context().native.active or context().startup.layers != 70:
        raise ValueError('DFlash cycle requires idle full target workers and no native serving plan')
    prompts = tuple(tuple(row) for row in plan['prompt_ids'])
    rows = len(prompts)
    cycles, extent = plan['cycles'], plan['verify_rows']
    if (not 1 <= rows <= 3 or type(cycles) is not int or not 1 <= cycles <= 16 or
            type(extent) is not int or not 2 <= extent <= 8 or
            any(not row or len(row) > 8192-cycles*extent-extent for row in prompts)):
        raise ValueError('Bounded DFlash smoke requests/context/cycles required')
    group = get_tp_group()
    def vote(ready):
        flag = torch.tensor([int(ready)], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group.cpu_group)
        return bool(flag.item())
    def reduce_sum(tensor):
        htcore.mark_step()
        dist.all_reduce(tensor, group=group.device_group)
        return tensor
    def gather_output(tensor):
        htcore.mark_step()
        outputs = [torch.empty_like(tensor) for _ in range(group.world_size)]
        dist.all_gather(outputs, tensor, group=group.device_group)
        return torch.cat(outputs, -1)
    draft = load_checkpoint(plan['draft_checkpoint'], runner.device, tp_rank=worker.rank,
                            tp_size=group.world_size, reduce_sum=reduce_sum, gather_output=gather_output)
    draft.fold_gqa = plan.get('fold_gqa', True)
    ids = tuple(str(i) for i in range(rows))
    capacities = {rid: len(prompt)+cycles*extent+extent for rid, prompt in zip(ids, prompts)}
    target = PackedTargetExecutor(runner, request_capacities=capacities,
                                 native_swa=plan['native_swa'], feature_layers=draft.spec.target_layers)
    coordinator = DFlashCycle(target, draft, ids)
    target.session.fold_gqa = draft.fold_gqa
    anchors = torch.empty(rows, device=runner.device, dtype=torch.int32)
    for offset in range(0, max(map(len, prompts)), 256):
        requests = []
        global_counts = [0]*rows
        finished = []
        for index, (rid, prompt) in enumerate(zip(ids, prompts)):
            chunk = prompt[offset:offset+256]
            if chunk:
                last = offset+len(chunk) == len(prompt)
                requests.append(RequestTokens(rid, chunk, offset, 'prefill', (len(chunk)-1,) if last else ()))
                global_counts[index] = len(chunk)
                if last:
                    finished.append(index)
        prefill_batch = TokenBatch(tuple(requests))
        prefill = target.execute(prefill_batch)
        coordinator.append_features(prefill, torch.tensor(global_counts, device=runner.device, dtype=torch.int32))
        target.commit(prefill, prefill_batch.query_lengths)
        if finished:
            anchors.index_copy_(0, torch.tensor(finished, device=runner.device), prefill.logits.argmax(-1).to(torch.int32))
    remaining = torch.full((rows,), cycles*extent+1, device=runner.device, dtype=torch.int32)
    arena = target.session
    target.session = BoundKVSession(arena, (extent,)*rows, max(capacities.values()),
                                    static_pages=plan.get('static_pages', True))
    target.session.fold_gqa = draft.fold_gqa
    label = plan.get('label', 'dflash-target-cycle')
    if type(label) is not str or not label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in label):
        raise ValueError('Cycle evidence label must be a simple local directory name')
    root = context().startup.run_dir/label
    root.mkdir(exist_ok=True)
    report = {'rank': worker.rank, 'pass': False, 'cycles': [], 'source_target_layers': 70,
              'drafter_layers': draft.spec.layers, 'drafter_block': draft.spec.block, 'verify_rows': extent,
              'proposal_hidden_positions': list(range(1, extent)),
              'static_page_bindings': target.session.static_pages,
              'fold_gqa_heads': draft.fold_gqa,
              'feature_layers': draft.spec.target_layers, 'context_lengths': [len(row) for row in prompts],
              'scope': 'actual checkpoint eager functional cycles with private KV; no native/HTTP TPS or answer-quality claim'}
    recorded = bool(plan.get('record_cycle', False))
    routes = None
    if plan.get('routes', False):
        from gaudi_kernels.serving.diagnostics.route_distribution import RouteDistribution
        routes = RouteDistribution(target.model)
        report['intrusive_route_snapshots'] = True
    recorder, binding = None, None
    if recorded and cycles < 3:
        raise ValueError('Recorded cycle needs eager warmup, capture and at least one replay')
    try:
        for step in range(cycles):
            starts = tuple(len(arena.committed[rid]) for rid in ids)
            schedule = QueryBatch(tuple(RequestQueries(rid, extent, start, 'verify') for rid, start in zip(ids, starts)))
            before = {name: tuple(t.index_select(0, torch.tensor(
                        tuple(s for slots in arena.committed.values() for s in slots), device=runner.device)).cpu().clone()
                        for t in tensors) for name, tensors in arena.caches.items()}
            began = time.perf_counter_ns()
            metadata = target.session.prepare(schedule)
            prepared_at = time.perf_counter_ns()
            execution = {'kind': 'eager-warmup' if recorded else 'eager'}
            if recorded and step == 1:
                from gaudi_kernels.serving.diagnostics.cycle_recorder import CycleRecorder
                binding = PackedInputBuffers(schedule, runner.device)
                recorder = CycleRecorder(coordinator, binding, metadata, root, worker.rank, vote)
                result = recorder.capture(schedule, metadata, anchors, remaining)
                execution = {'kind': 'capture', **recorder.record}
            elif recorded and step > 1:
                result, timing = recorder.replay(schedule, metadata,
                                                 profile=bool(plan.get('profile', False) and step == 4))
                execution = {'kind': 'recorded-replay', **timing}
            else:
                result = coordinator.run_prepared(schedule, metadata, anchors, remaining)
            executed_at = time.perf_counter_ns()
            output = coordinator.finish_at_output_boundary(result)
            delivered_at = time.perf_counter_ns()
            elapsed = delivered_at-began
            emitted = tuple(len(row.emitted_tokens) for row in output.requests)
            # Keep external input addresses stable for the recorded cycle.
            anchors.copy_(result.next_anchor_ids)
            remaining.sub_(result.verification.emitted_counts)
            old_slots = tuple(s for rid, start in zip(ids, starts) for s in arena.committed[rid][:start])
            index = torch.tensor(old_slots, device=runner.device, dtype=torch.int64)
            unchanged = all(torch.equal(saved.contiguous().view(torch.uint8),
                tensor.index_select(0, index).cpu().contiguous().view(torch.uint8))
                for name, tensors in arena.caches.items() for saved, tensor in zip(before[name], tensors))
            record = {'step': step, 'starts': starts, 'emitted': emitted,
                      'execution': execution,
                      # These readbacks are outside the measured device chain,
                      # after output delivery. Keep proposal/target alignment
                      # visible; acceptance alone cannot diagnose bad indexing.
                      'proposed_ids': result.proposed_ids.cpu().tolist(),
                      'target_next_ids': result.target.logits.argmax(-1).reshape(rows, extent).cpu().tolist(),
                      'emitted_ids': [list(row.emitted_tokens) for row in output.requests],
                      'matched_drafts': result.verification.matched_draft_counts.cpu().tolist(),
                      'eager_cycle_wall_ns': elapsed, 'old_target_kv_bytes_unchanged': unchanged,
                      'prepare_wall_ns': prepared_at-began,
                      'execute_wall_ns': executed_at-prepared_at,
                      'delivery_wall_ns': delivered_at-executed_at,
                      'finite': bool(torch.isfinite(result.target.hidden).all().cpu() and
                                     torch.isfinite(result.target.logits).all().cpu())}
            _, ring_positions, ring_valid = coordinator.context.state()
            actual_positions = ring_positions.cpu()
            actual_valid = ring_valid.cpu()
            record['committed_context_positions_correct'] = all(
                sorted(actual_positions[i][actual_valid[i]].tolist()) ==
                list(range(max(0, start+n-draft.spec.window), start+n))
                for i, (start, n) in enumerate(zip(starts, emitted)))
            record['pass'] = (unchanged and record['finite'] and all(1 <= n <= extent for n in emitted) and
                              record['committed_context_positions_correct'] and
                              all(len(arena.committed[rid]) == start+n for rid, start, n in zip(ids, starts, emitted)))
            if routes is not None:
                record['route_distribution'] = routes.summarize(schedule)
            report['cycles'].append(record)
            if not record['pass']:
                raise RuntimeError('DFlash cycle finite/live-KV/input-commit check failed')
            (root/f'rank{worker.rank}.json').write_text(json.dumps(report, indent=2)+'\n')
        report['mean_emitted_per_request_cycle'] = sum(sum(r['emitted']) for r in report['cycles'])/(rows*cycles)
        report['pass'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        if routes is not None:
            routes.close()
        if recorder is not None:
            report['release_code'] = recorder.release()
            if report['release_code']:
                report['pass'] = False
        (root/f'rank{worker.rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
