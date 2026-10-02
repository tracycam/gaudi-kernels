"""Optional resident batch ABBA, with no device/framework imports.

The caller supplies the existing policy/drain RPC and streaming generation.
Every arm starts with the same complete bridge preconditioning sequence.
"""
import statistics
from tools.validation.legacy_selection import typed_policy
import time


def prefill_completion_window(steps, prompt_tokens_each):
    """Coordinator start through every request's first emitted token.

    This includes scheduler stalls and any early requests already decoding.
    It is serving prompt completion throughput, never isolated MME throughput.
    """
    batch = len(prompt_tokens_each)
    if not batch or any(n <= 0 for n in prompt_tokens_each):
        raise ValueError('Positive prompt lengths required')
    finish = next((i for i, s in enumerate(steps)
                   if s.get('known_requests') == batch and s.get('emitted_min', 0) >= 1), None)
    if finish is None:
        return dict(qualified=False, reason='not every request emitted its first token')
    window = steps[:finish + 1]
    seconds = (window[-1]['end_ns'] - window[0]['begin_ns']) / 1e9
    if seconds <= 0:
        return dict(qualified=False, reason='invalid coordinator interval')
    generated = sum(s['new_tokens'] for s in window)
    return dict(qualified=True, prompt_tokens=sum(prompt_tokens_each),
                duration_s=seconds, prompt_completion_tps=sum(prompt_tokens_each) / seconds,
                generated_tokens_in_interval=generated, early_decode_tokens=max(0, generated - batch),
                steps=len(window), stall_steps=sum(s['new_tokens'] == 0 for s in window),
                scope='coordinator start to all requests first-token completion, including stalls and interleaved decode; not pure prefill compute or model acceptance')


def validate_batch_abba(control, policies, batches, expect_bitwise=False):
    if not control:
        if expect_bitwise:
            raise ValueError('Batch cross-policy bitwise gate requires --batch-policy-abba-control')
        return
    if not batches or not policies or any(p is None for p in policies):
        raise ValueError('Batch policy ABBA requires batch sizes and explicit resident production policies')
    if control not in policies or policies[-1] == control:
        raise ValueError('Batch policy ABBA needs a resident control and a distinct final candidate')
    if len(set(batches)) != len(batches):
        raise ValueError('Batch policy ABBA requires unique batch sizes')


def matured_batch_window(steps, batch, token_limit):
    """Keep every step between mature simultaneous decoding and first completion.

    A zero-output coordinator step is a stall, not a reason to splice windows.
    Count actual returned tokens rather than multiplying batch by step count.
    All raw steps remain in the caller's row, including cold/prefill/completion.
    """
    indices = [i for i, s in enumerate(steps)
               if s.get('known_requests') == batch and s.get('emitted_min', 0) >= 32
               and s['emitted'] < token_limit]
    if not indices:
        return dict(qualified=False, reason='no matured simultaneous batch window', steps=0)
    window = steps[indices[0]:indices[-1] + 1]
    if indices != list(range(indices[0], indices[-1] + 1)):
        return dict(qualified=False, reason='noncontiguous mature batch state', steps=len(window))
    for s in window:
        increments = s.get('per_request_new', [])
        if len(increments) > batch or any(n != 1 for n in increments) or s['new_tokens'] != sum(increments):
            return dict(qualified=False, reason='unexpected cumulative token accounting', steps=len(window))
    tokens = sum(s['new_tokens'] for s in window)
    seconds = (window[-1]['end_ns'] - window[0]['begin_ns']) / 1e9
    if len(window) < 16 or tokens < 16 * batch or seconds <= 0:
        return dict(qualified=False, reason='insufficient complete mature decode work', steps=len(window),
                    new_tokens=tokens)
    return dict(qualified=True, actual_batch=batch, steps=len(window), new_tokens=tokens,
                duration_s=seconds, aggregate_decode_tps=tokens / seconds,
                median_step_ms=statistics.median(s['ms'] for s in window),
                first_emitted_min=window[0]['emitted_min'], last_emitted_max=window[-1]['emitted'],
                selected_step_indices=indices,
                stall_steps=sum(s['new_tokens'] == 0 for s in window),
                partial_emission_steps=sum(0 < s['new_tokens'] < batch for s in window),
                scope='all contiguous matured full-batch steps including stalls and partial emissions; actual token count / full wall interval')


def run_batch_policy_abba(*, rpc, generate, make_prompts, policies, control, batches,
                          token_limit, layers, expect_bitwise, steps, counts, result, save,
                          capture_gate=None):
    """Populate result incrementally so failures retain their exact arm/fixtures.

    generate(prompts, token_limit) returns vLLM-shaped RequestOutputs and updates
    the existing coordinator steps/counts. No policy names are interpreted here.
    """
    validate_batch_abba(control, policies, batches, expect_bitwise)
    candidate = policies[-1]
    metadata = dict(status='RUNNING', control=control, candidate=candidate,
                    order=[control, candidate, candidate, control],
                    expect_cross_policy_bitwise=expect_bitwise, backend='bridge',
                    preconditioning=[], transitions=[], comparisons=[],
                    scope='same resident model; full identical prompt/token warmup before every arm; cold work excluded and retained')
    result['batch_policy_abba'] = metadata
    result['batch_runs'] = []
    save()
    for batch in batches:
        prompts, codes = make_prompts(batch)
        if len(prompts) != batch or len(codes) != batch:
            raise ValueError('Batch prompt/code cardinality mismatch')
        baselines = {}
        for arm, policy in enumerate(metadata['order']):
            # Even B2 repeats the full drain/cache-clear lifecycle, so it cannot
            # inherit a different warm state from B1. Both calls synchronize in
            # their existing implementation; never reset between warm/measure.
            controls = rpc('native_configure', args=(False,))
            changed = rpc('production_policy', args=(typed_policy(policy),))
            transition = dict(batch=batch, arm=arm, policy=policy, controls=controls, changes=changed)
            metadata['transitions'].append(transition)
            save()
            if len(controls) != 8 or any(x.get('active') is not False for x in controls):
                raise AssertionError('Batch ABBA failed to disable native replay on every rank')
            if len(changed) != 8 or any(x['details']['selected_layer_count'] != layers for x in changed):
                raise AssertionError('Batch ABBA policy did not select every requested QKV layer on all ranks')
            for trial in ('warmup', 'measured'):
                steps.clear()
                counts.clear()
                begin = time.monotonic()
                outputs = generate(prompts, token_limit)
                elapsed = time.monotonic() - begin
                texts = [x.outputs[0].text if x.outputs else '' for x in outputs]
                ids = [list(x.outputs[0].token_ids) if x.outputs else [] for x in outputs]
                complete = len(ids) == batch and all(len(x) == token_limit for x in ids)
                quality = len(texts) == batch and all(code in text for code, text in zip(codes, texts))
                same = policy not in baselines or ids == baselines[policy]
                cross = control not in baselines or ids == baselines[control]
                if policy not in baselines and complete:
                    baselines[policy] = ids
                native = rpc('native_summary')
                bridge_only = len(native) == 8 and all(rank.get('native_steps') == [] for rank in native)
                row = dict(batch=batch, arm=arm, trial=trial, policy=policy, active=False,
                           prompt_token_ids=[list(p['prompt_token_ids']) for p in prompts],
                           prompt_tokens_each=[len(p['prompt_token_ids']) for p in prompts],
                           generated_tokens_each=token_limit, wall_s=elapsed, texts=texts, ids=ids,
                           token_count_pass=complete, quality_pass=quality, repeat_match=same,
                           cross_policy_token_match=cross, cross_policy_match_required=expect_bitwise,
                           bridge_only=bridge_only, ranks=native,
                           steps=list(steps), full_batch_decode_timing=matured_batch_window(steps, batch, token_limit),
                           prefill_completion_timing=prefill_completion_window(steps, [len(p['prompt_token_ids']) for p in prompts]),
                           production_state=rpc('production_snapshot'),
                           performance_qualified=False, model_candidate_accepted=False,
                           scope='complete bridge warmup outside timing' if trial == 'warmup'
                           else 'warmed resident bridge policy ABBA; no warmup/outlier/stall subtraction')
                row['execution_gate_pass'] = complete and quality and same and bridge_only and (cross or not expect_bitwise)
                if capture_gate is not None:
                    row['capture_gate'] = capture_gate(policy, batch, row['production_state'])
                    row['execution_gate_pass'] &= row['capture_gate']['pass']
                destination = metadata['preconditioning'] if trial == 'warmup' else result['batch_runs']
                destination.append(row)
                save()
                if not row['execution_gate_pass']:
                    metadata['status'] = 'REJECTED'
                    save()
                    raise AssertionError(f'Batch policy ABBA output gate failed: B{batch} arm {arm} {trial}')
        measured = [r for r in result['batch_runs'] if r['batch'] == batch]
        pooled = {}
        for label, policy in (('control', control), ('candidate', candidate)):
            rows = [r for r in measured if r['policy'] == policy]
            windows = [r['full_batch_decode_timing'] for r in rows]
            prefill = [r['prefill_completion_timing'] for r in rows]
            qualified = len(rows) == 2 and all(w.get('qualified') for w in windows)
            pooled[label] = dict(policy=policy, arms=[r['arm'] for r in rows], timing_windows_qualified=qualified,
                                 aggregate_decode_tps=(sum(w['new_tokens'] for w in windows) /
                                                       sum(w['duration_s'] for w in windows)) if qualified else None)
            pooled[label]['prefill_completion_tps'] = (sum(w['prompt_tokens'] for w in prefill) /
                sum(w['duration_s'] for w in prefill)) if len(prefill) == 2 and all(w.get('qualified') for w in prefill) else None
        metadata['comparisons'].append(dict(batch=batch, pooled=pooled,
                                           cross_policy_token_match=baselines[control] == baselines[candidate],
                                           performance_qualified=False))
        save()
    # Preserve the existing runner's final-policy state for lifecycle/long
    # context work, outside all timed arms. The fourth measured arm remains A.
    controls = rpc('native_configure', args=(False,))
    changed = rpc('production_policy', args=(typed_policy(candidate),))
    metadata['restore_final_policy'] = dict(policy=candidate, controls=controls, changes=changed)
    save()
    if (len(controls) != 8 or any(x.get('active') is not False for x in controls)
            or len(changed) != 8 or any(x['details']['selected_layer_count'] != layers for x in changed)):
        raise AssertionError('Batch ABBA final-policy restoration failed')
    metadata['status'] = 'EXECUTION_PASS_MODEL_QUALITY_PENDING'
    save()


def finalize_batch_abba(result, layers):
    """Only the caller's unchanged full-model quality decision can qualify TPS."""
    metadata = result.get('batch_policy_abba')
    if metadata is None:
        return
    accepted = layers == 70 and result.get('candidate_accepted') is True
    for row in metadata['preconditioning']:
        row['model_candidate_accepted'] = accepted
        row['performance_qualified'] = False  # Cold preconditioning is never measured ABBA.
    for row in result['batch_runs']:
        row['model_candidate_accepted'] = accepted
        row['performance_qualified'] = bool(accepted and row['execution_gate_pass']
                                             and row['full_batch_decode_timing'].get('qualified'))
    for comparison in metadata['comparisons']:
        comparison['performance_qualified'] = accepted and all(
            r['performance_qualified'] for r in result['batch_runs'] if r['batch'] == comparison['batch'])
    metadata['model_candidate_accepted'] = accepted
    metadata['status'] = ('PASS' if all(c['performance_qualified'] for c in metadata['comparisons'])
                          else 'MODEL_ACCEPTED_TIMING_UNQUALIFIED') if accepted else 'MODEL_REJECTED_OR_TRUNCATED'
