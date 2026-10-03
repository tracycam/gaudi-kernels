"""Bounded real-model packed target diagnostics; never timed serving."""
import json


def on_worker(worker, plan):
    import torch
    from gaudi_kernels.engine.context import context
    from gaudi_kernels.engine.token_batch import BatchCapacity, RequestTokens, TokenBatch
    from gaudi_kernels.serving.executor.packed_target import PackedTargetExecutor
    from gaudi_kernels.serving.executor.production_quality import compare_logits

    runner = worker.model_runner
    if runner.input_batch.num_reqs:
        raise ValueError('Diagnostic target arena requires no active scheduled requests')
    report = {'rank': worker.rank, 'cases': [], 'scope': 'actual loaded target, exclusive KV arena; no native TPS claim'}
    vocabulary = runner.vllm_config.model_config.hf_text_config.vocab_size
    for case in plan['cases']:
        rows = case['rows']
        starts = case['starts']
        capacity = sum(starts) + sum(rows) + 32
        options = (dict(request_capacities={str(i): starts[i] + rows[i] + 32 for i in range(len(rows))},
                        native_swa=True) if plan.get('native_swa') else dict(kv_capacity=capacity))
        packed = PackedTargetExecutor(runner, **options)
        sequential = PackedTargetExecutor(runner, **options)
        histories = []
        for index, start in enumerate(starts):
            tokens = tuple(case['prefix_tokens'][index]) if 'prefix_tokens' in case else tuple(
                (11 + index * 17 + p * 13) % vocabulary for p in range(start))
            histories.append(tokens)
            if start:
                batch = TokenBatch((RequestTokens(str(index), tokens, 0, 'prefill'),))
                for executor in (packed, sequential):
                    value = executor.execute(batch)
                    executor.commit(value, (start,))
        requests = tuple(RequestTokens(str(i), tuple(case['query_tokens'][i]) if 'query_tokens' in case else
                         tuple((101 + i * 19 + p * 7) % vocabulary for p in range(n)),
                         starts[i], 'decode' if n == 1 else ('verify' if i == 1 else 'prefill'), tuple(range(n)))
                         for i, n in enumerate(rows))
        batch = TokenBatch(requests, BatchCapacity(sum(rows) + 11, len(rows) + 1, sum(rows) + 1))
        trace = None
        if plan.get('trace_layers') and case['name'] == 'serving-teacher':
            from gaudi_kernels.serving.diagnostics.packed_boundaries import BoundaryTrace, compare_boundaries
            trace = BoundaryTrace(packed.model, plan['trace_layers'])
            trace.begin(batch.num_tokens)
        original_slots = {request.request_id: packed.session.committed.get(request.request_id, ())
                          for request in requests}
        live_indices = torch.tensor(tuple(s for slots in original_slots.values() for s in slots),
                                    device=runner.device, dtype=torch.int64)
        past_storage = {name: tuple(tensor.index_select(0, live_indices).cpu() for tensor in tensors)
                        for name, tensors in packed.session.caches.items()}
        calls = []
        handles = []
        def hook(name):
            def observe(module, args):
                calls.append({'module': name, 'rows': args[0].shape[0]})
            return observe
        for name, module in packed.model.named_modules():
            if name.endswith(('qkv_proj', 'o_proj', 'moe_op')):
                handles.append(module.register_forward_pre_hook(hook(name)))
        from gaudi_kernels.production_integration import snapshot as block_snapshot
        before_branches = block_snapshot()['python_apply_branch_counts']
        try:
            candidate = packed.execute(batch)
            actual_hidden = candidate.hidden.detach().cpu().clone()
            actual_logits = candidate.logits.detach().cpu().clone()
            candidate_boundaries = trace.freeze() if trace else None
        finally:
            for handle in handles:
                handle.remove()
        after_branches = block_snapshot()['python_apply_branch_counts']
        branch_delta = {key: value - before_branches.get(key, 0) for key, value in after_branches.items()
                        if value != before_branches.get(key, 0)}
        reference_hidden, reference_logits, reference_boundaries = [], [], []
        for request in requests:
            for local, token in enumerate(request.token_ids):
                single = TokenBatch((RequestTokens(request.request_id, (token,), request.start_position + local,
                                                   'decode', (0,)),))
                if trace:
                    trace.begin(1)
                output = sequential.execute(single)
                if trace:
                    reference_boundaries.append(trace.freeze())
                sequential.commit(output, (1,))
                reference_hidden.append(output.hidden.cpu())
                reference_logits.append(output.logits.cpu())
        expected_hidden = torch.cat(reference_hidden)
        expected_logits = torch.cat(reference_logits)
        checks = [compare_logits(expected_logits[i:i+1], actual_logits[i:i+1], contract='fp32_arithmetic_v1')
                  for i in range(sum(rows))]
        delta = actual_hidden.float() - expected_hidden.float()
        row = {'name': case['name'], 'rows': rows, 'valid_rows': batch.num_tokens,
               'capacity_rows': batch.capacity.token_rows, 'output_owners': candidate.output_owners,
               'calls': calls, 'checks': checks,
               'qkv_arithmetic_branches': branch_delta,
               'hidden_relative_l2': (delta.norm() / expected_hidden.float().norm()).item(),
               'hidden_max_abs': delta.abs().max().item(),
               'all_query_pass': all(check['pass'] for check in checks),
               'every_shared_call_compact': bool(calls) and all(call['rows'] == sum(rows) for call in calls)}
        if trace:
            trace.close()
            row['operator_boundaries'] = compare_boundaries(candidate_boundaries, reference_boundaries)
            root = context().startup.run_dir/'packed-operator-boundaries'
            root.mkdir(exist_ok=True)
            torch.save({'candidate': candidate_boundaries, 'sequential': reference_boundaries},
                       root/f'rank{worker.rank}.pt')
        if 'serving_rows' in case:
            cache = {}
            teacher_checks = []
            for index, source in enumerate(case['serving_rows']):
                if source['path'] not in cache:
                    cache[source['path']] = torch.load(source['path'], weights_only=True, map_location='cpu')['logits']
                baseline = cache[source['path']][source['row']:source['row']+1]
                teacher_checks.append(compare_logits(baseline, actual_logits[index:index+1],
                                                      contract='fp32_arithmetic_v1'))
            row['serving_teacher_checks'] = teacher_checks
            row['serving_teacher_pass'] = all(check['pass'] for check in teacher_checks)
        # Mutate only future verify queries; neither prior logits nor other
        # requests may change. This detects masks and cross-row quantization.
        packed.abort(candidate)
        changed = list(requests)
        request = changed[1]
        changed[1] = RequestTokens(request.request_id,
            tuple(token if local < 2 else (token + 307) % vocabulary
                  for local, token in enumerate(request.token_ids)),
            request.start_position, request.kind, request.output_rows)
        mutated = packed.execute(TokenBatch(tuple(changed), batch.capacity))
        mutated_logits = mutated.logits.cpu()
        unaffected = list(range(batch.num_tokens))
        for local in range(2, request.query_length):
            unaffected.remove(batch.query_start_loc[1] + local)
        row['future_request_isolation_exact'] = torch.equal(actual_logits[unaffected], mutated_logits[unaffected])
        packed.abort(mutated)
        candidate = packed.execute(batch)
        row['live_history_bytes_unchanged'] = all(torch.equal(saved.contiguous().view(torch.uint8),
            tensor.index_select(0, live_indices).cpu().contiguous().view(torch.uint8))
            for name, tensors in packed.session.caches.items() for saved, tensor in zip(past_storage[name], tensors))
        # Reject candidate suffixes; retain the independent sequential oracle's
        # original history and only its accepted rows. Rebuilding the whole
        # history as one prefill changes QKV A8/A16 and is not a KV oracle.
        prefixes = case['commits']
        packed.commit(candidate, tuple(prefixes))
        continuation = sequential
        for request, count in zip(requests, prefixes):
            slots = continuation.session.committed[request.request_id]
            continuation.session.committed[request.request_id] = slots[:request.start_position + count]
        row['continuation_reference'] = 'original history plus token-at-a-time accepted prefix; no history re-quantization'
        row['commit_slot_coverage_exact'] = all(
            packed.session.committed[request.request_id] == (*original_slots[request.request_id],
                                                           *candidate.transaction.candidate_slots[index][:count])
            for index, (request, count) in enumerate(zip(requests, prefixes)))
        following = TokenBatch(tuple(RequestTokens(str(i), (211 + i,), starts[i] + prefixes[i], 'decode')
                                     for i in range(len(rows))))
        kv_differences = []
        for name, tensors in packed.session.caches.items():
            reference_tensors = continuation.session.caches[name]
            for request, count in zip(requests, prefixes):
                packed_slots = packed.session.committed[request.request_id][request.start_position:]
                oracle_slots = continuation.session.committed[request.request_id][request.start_position:]
                if not count:
                    continue
                a_indices = torch.tensor(packed_slots, dtype=torch.int64, device=runner.device)
                b_indices = torch.tensor(oracle_slots, dtype=torch.int64, device=runner.device)
                for kind, tensor, reference_tensor in zip(('key', 'value'), tensors, reference_tensors):
                    a = tensor.index_select(0, a_indices).cpu().float()
                    b = reference_tensor.index_select(0, b_indices).cpu().float()
                    if not torch.equal(a, b):
                        kv_differences.append({'layer': name, 'request': request.request_id, 'kind': kind,
                            'relative_l2': ((a-b).norm()/b.norm().clamp_min(1e-30)).item(),
                            'max_abs': (a-b).abs().max().item()})
        row['accepted_kv_differences'] = kv_differences
        actual_next = packed.execute(following)
        actual_next_logits = actual_next.logits.detach().cpu().clone()
        expected_next = continuation.execute(following)
        expected_next_logits = expected_next.logits.detach().cpu().clone()
        next_checks = [compare_logits(expected_next_logits[i:i+1], actual_next_logits[i:i+1],
                       contract='fp32_arithmetic_v1') for i in range(len(rows))]
        row['continuation_checks'] = next_checks
        row['continuation_pass'] = all(check['pass'] for check in next_checks)
        packed.abort(actual_next)
        continuation.abort(expected_next)
        directory = context().startup.run_dir / 'packed-target'
        directory.mkdir(exist_ok=True)
        if worker.rank == 0:
            torch.save({'candidate_hidden': actual_hidden, 'reference_hidden': expected_hidden,
                        'candidate_logits': actual_logits, 'reference_logits': expected_logits,
                        'continuation_logits': actual_next_logits,
                        'continuation_reference_logits': expected_next_logits}, directory/(case['name']+'.pt'))
        report['cases'].append(row)
        (directory/f'rank{worker.rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    report['pass'] = all(c['all_query_pass'] and c['continuation_pass'] and c['every_shared_call_compact']
                         and c['future_request_isolation_exact']
                         and c['commit_slot_coverage_exact']
                         and c['live_history_bytes_unchanged']
                         and c.get('serving_teacher_pass', True)
                         for c in report['cases'])
    (directory/f'rank{worker.rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
