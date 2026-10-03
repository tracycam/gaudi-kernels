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
        packed = PackedTargetExecutor(runner, kv_capacity=capacity)
        sequential = PackedTargetExecutor(runner, kv_capacity=capacity)
        histories = []
        for index, start in enumerate(starts):
            tokens = tuple((11 + index * 17 + p * 13) % vocabulary for p in range(start))
            histories.append(tokens)
            if start:
                batch = TokenBatch((RequestTokens(str(index), tokens, 0, 'prefill'),))
                for executor in (packed, sequential):
                    value = executor.execute(batch)
                    executor.commit(value, (start,))
        requests = tuple(RequestTokens(str(i), tuple((101 + i * 19 + p * 7) % vocabulary for p in range(n)),
                         starts[i], 'decode' if n == 1 else ('verify' if i == 1 else 'prefill'), tuple(range(n)))
                         for i, n in enumerate(rows))
        batch = TokenBatch(requests, BatchCapacity(sum(rows) + 11, len(rows) + 1, sum(rows) + 1))
        calls = []
        handles = []
        def hook(name):
            def observe(module, args):
                calls.append({'module': name, 'rows': args[0].shape[0]})
            return observe
        for name, module in packed.model.named_modules():
            if name.endswith(('qkv_proj', 'o_proj', 'moe_op')):
                handles.append(module.register_forward_pre_hook(hook(name)))
        try:
            candidate = packed.execute(batch)
        finally:
            for handle in handles:
                handle.remove()
        reference_hidden, reference_logits = [], []
        for request in requests:
            for local, token in enumerate(request.token_ids):
                single = TokenBatch((RequestTokens(request.request_id, (token,), request.start_position + local,
                                                   'decode', (0,)),))
                output = sequential.execute(single)
                sequential.commit(output, (1,))
                reference_hidden.append(output.hidden.cpu())
                reference_logits.append(output.logits.cpu())
        expected_hidden = torch.cat(reference_hidden)
        expected_logits = torch.cat(reference_logits)
        actual_hidden, actual_logits = candidate.hidden.cpu(), candidate.logits.cpu()
        checks = [compare_logits(expected_logits[i:i+1], actual_logits[i:i+1], contract='fp32_arithmetic_v1')
                  for i in range(sum(rows))]
        delta = actual_hidden.float() - expected_hidden.float()
        row = {'name': case['name'], 'rows': rows, 'valid_rows': batch.num_tokens,
               'capacity_rows': batch.capacity.token_rows, 'output_owners': candidate.output_owners,
               'calls': calls, 'checks': checks,
               'hidden_relative_l2': (delta.norm() / expected_hidden.float().norm()).item(),
               'hidden_max_abs': delta.abs().max().item(),
               'all_query_pass': all(check['pass'] for check in checks),
               'every_shared_call_compact': bool(calls) and all(call['rows'] == sum(rows) for call in calls)}
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
        # Reject candidate suffixes; compare continuation with a fresh committed reference.
        prefixes = case['commits']
        packed.commit(candidate, tuple(prefixes))
        continuation = PackedTargetExecutor(runner, kv_capacity=capacity)
        for index, (past, request, count) in enumerate(zip(histories, requests, prefixes)):
            tokens = (*past, *request.token_ids[:count])
            if tokens:
                value = continuation.execute(TokenBatch((RequestTokens(str(index), tokens, 0, 'prefill'),)))
                continuation.commit(value, (len(tokens),))
        following = TokenBatch(tuple(RequestTokens(str(i), (211 + i,), starts[i] + prefixes[i], 'decode')
                                     for i in range(len(rows))))
        actual_next = packed.execute(following)
        expected_next = continuation.execute(following)
        next_checks = [compare_logits(expected_next.logits[i:i+1].cpu(), actual_next.logits[i:i+1].cpu(),
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
                        'continuation_logits': actual_next.logits.cpu(),
                        'continuation_reference_logits': expected_next.logits.cpu()}, directory/(case['name']+'.pt'))
        report['cases'].append(row)
        (directory/f'rank{worker.rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    report['pass'] = all(c['all_query_pass'] and c['continuation_pass'] and c['every_shared_call_compact']
                         and c['future_request_isolation_exact']
                         for c in report['cases'])
    (directory/f'rank{worker.rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    return report

