"""vLLM graph boundary adapter; all execution remains in the PyTorch bridge.

The first admitted contract is synchronous, penalty-free greedy decode. Request
state updates, scheduler decisions and output handling remain in the base runner.
"""


def plain_greedy(params):
    if params is None or params.temperature != 0:
        return False
    if (params.repetition_penalty != 1 or params.presence_penalty != 0 or
            params.frequency_penalty != 0 or params.min_tokens != 0):
        return False
    # Even a zero-logprob request has different output semantics from None.
    if params.logprobs is not None or params.prompt_logprobs is not None:
        return False
    if getattr(params, 'thinking_token_budget', None) is not None:
        return False
    return not any(getattr(params, key, None) for key in (
        'allowed_token_ids', 'bad_words', 'bad_words_token_ids', 'logits_processors', 'logit_bias',
        'structured_outputs', 'logprob_token_ids'))


def _has_speculative_tokens(runner):
    spec = getattr(runner, 'speculative_config', None)
    count = getattr(spec, 'num_speculative_tokens', 0)
    return isinstance(count, int) and not isinstance(count, bool) and count > 0


def admitted(runner, metadata, token_ids, warmup_mode, lora_logits_mask, lora_mask):
    if any(getattr(runner, flag, None) for flag in (
            '_production_boundary_session', '_production_cpu_reference',
            '_production_same_input_audit', '_production_layer_diagnostics')):
        return False
    spec = getattr(runner, 'speculative_config', None)
    speculative = _has_speculative_tokens(runner)
    if (warmup_mode or getattr(metadata, 'is_prompt', True) or
            (spec is not None and not speculative) or runner.use_async_scheduling or
            runner.use_structured_output or
            (runner.use_aux_hidden_state_outputs and not speculative) or
            lora_logits_mask is not None or lora_mask is not None):
        return False
    parallel = runner.vllm_config.parallel_config
    if (parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1 or
            runner.vllm_config.lora_config is not None or
            runner.vllm_config.kv_transfer_config is not None or
            getattr(runner.vllm_config, 'max_concurrent_batches', 1) != 1):
        return False
    batch = runner.input_batch
    # A custom processor can depend on CPU request state even under greedy
    # sampling. Only known inactive builtins can safely disappear at capture.
    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor, MinTokensLogitsProcessor)
    for processor in batch.logitsprocs.non_argmax_invariant:
        if type(processor) is LogitBiasLogitsProcessor and not processor.biases:
            continue
        if type(processor) is MinTokensLogitsProcessor and not processor.min_toks:
            continue
        return False
    holder = getattr(batch, 'thinking_budget_state_holder', None)
    if holder is not None and holder.has_tracked_requests():
        return False
    if (batch.num_reqs < 1 or getattr(runner, '_step_decode_count', 0) != batch.num_reqs or
            token_ids.ndim != 2 or token_ids.shape[1] != 1 or
            token_ids.shape[0] < batch.num_reqs or token_ids.shape[0] % batch.num_reqs != 0):
        return False
    if not speculative and token_ids.shape[0] != batch.num_reqs:
        return False
    return all(plain_greedy(runner.requests[rid].sampling_params)
               for rid in batch.req_ids[:batch.num_reqs])


def execute(runner, original, token_ids, position_ids, metadata, logits_indices,
            kv_caches, lora_logits_mask, lora_mask, warmup_mode=False,
            inputs_embeds=None, model_mm_kwargs=None):
    import habana_frameworks.torch as ht
    from vllm.v1.outputs import SamplerOutput

    runner._step_graph_sample = None
    runner._step_graph_spec_sample = None
    if (inputs_embeds is not None or model_mm_kwargs or kv_caches is not runner.kv_caches or
            not admitted(runner, metadata, token_ids, warmup_mode, lora_logits_mask, lora_mask)):
        return original(token_ids, position_ids, metadata, logits_indices, kv_caches,
                        lora_logits_mask, lora_mask, warmup_mode, inputs_embeds, model_mm_kwargs)

    batch = runner.input_batch
    speculative = token_ids.shape[0] > batch.num_reqs
    model_holder = getattr(runner, 'model', None)
    unwrapped_model = getattr(model_holder, 'model', model_holder)
    has_shard_lm_head = (
        unwrapped_model is not None
        and getattr(unwrapped_model, 'lm_head', None) is not None
        and getattr(unwrapped_model, 'logits_processor', None) is not None
    )
    skip_full_logits = bool(
        speculative
        and getattr(runner, '_skip_full_logits_in_spec_graph', has_shard_lm_head)
    )
    key = (token_ids.shape[0], logits_indices.numel(), batch.num_reqs,
           getattr(runner, '_production_selected_policy', None), speculative)
    if skip_full_logits:
        key = (*key, True)
    programs = runner.__dict__.setdefault('_step_graph_programs', {})
    if not speculative:
        # Sampling tensors are not read by this admitted greedy contract. Build the
        # original metadata once per row layout; never reimplement the sampler.
        if key not in programs:
            import weakref
            request_ids = list(batch.req_ids[:batch.num_reqs])
            sampling = runner._prepare_sampling(runner.batch_changed, request_ids, logits_indices.numel())
            if not sampling.all_greedy or not sampling.no_penalties:
                raise RuntimeError('Greedy graph admission disagrees with sampler metadata')

            runner_ref, model_program = weakref.ref(runner), original.__func__

            def tensor_program(tokens, positions, attn_metadata, indices):
                owner = runner_ref()
                if owner is None:
                    raise RuntimeError('Step graph outlived its model runner')
                owner._inside_step_graph = True
                try:
                    # HpuModelAdapter removes this legacy argument; attention uses
                    # its bound module buffers. Do not hash/copy that unused tree
                    # on every token. KV reinitialization invalidates this program.
                    values = model_program(owner, tokens, positions, attn_metadata, indices,
                                           owner.kv_caches, None, None)
                    sample = owner.sampler(logits=values[3], sampling_metadata=sampling)
                    if sample.logprobs_tensors is not None:
                        raise RuntimeError('Admitted greedy graph unexpectedly produced logprobs')
                    return (*values, sample.sampled_token_ids)
                finally:
                    owner._inside_step_graph = False

            programs[key] = (ht.hpu.wrap_in_hpu_graph_func(
                tensor_program, disable_tensor_cache=False, asynchronous=False), sampling)
        program, sampling = programs[key]
        values = program(token_ids, position_ids, metadata, logits_indices)
        runner._step_graph_sample = (values[3], SamplerOutput(
            sampled_token_ids=values[4], logprobs_tensors=None), sampling)
        runner._step_graph_calls = getattr(runner, '_step_graph_calls', 0) + 1
        return values[:4]

    if key not in programs:
        import weakref
        sampling = None
        if hasattr(runner, '_prepare_sampling'):
            request_ids = list(batch.req_ids[:batch.num_reqs])
            try:
                sampling = runner._prepare_sampling(runner.batch_changed, request_ids, logits_indices.numel())
            except TypeError:
                sampling = runner._prepare_sampling(runner.batch_changed, request_ids)
            if not sampling.all_greedy or not sampling.no_penalties:
                raise RuntimeError('Greedy graph admission disagrees with sampler metadata')
        runner_ref, model_program = weakref.ref(runner), original.__func__
        unset = object()

        def spec_tensor_program(tokens, positions, attn_metadata, indices):
            owner = runner_ref()
            if owner is None:
                raise RuntimeError('Step graph outlived its model runner')
            owner._inside_step_graph = True
            target_holder = getattr(owner, 'model', None)
            saved_compute_logits = unset
            if (
                skip_full_logits
                and target_holder is not None
                and hasattr(target_holder, 'compute_logits')
            ):
                saved_compute_logits = target_holder.__dict__.get('compute_logits', unset)
                target_holder.compute_logits = lambda *a, **kw: None
            try:
                values = model_program(owner, tokens, positions, attn_metadata, indices,
                                       owner.kv_caches, None, None)
            finally:
                if saved_compute_logits is not unset or (
                    skip_full_logits
                    and target_holder is not None
                    and 'compute_logits' in getattr(target_holder, '__dict__', {})
                ):
                    if saved_compute_logits is unset:
                        target_holder.__dict__.pop('compute_logits', None)
                    else:
                        target_holder.compute_logits = saved_compute_logits
            try:
                logits = values[3]
                if logits is None and getattr(owner, 'model', None) is not None and values[2] is not None:
                    from gaudi_kernels.serving.output_reduction import compute_greedy_tokens
                    greedy_tokens = compute_greedy_tokens(owner.model, values[2])
                elif hasattr(logits, 'argmax'):
                    greedy_tokens = logits.argmax(-1)
                    if hasattr(greedy_tokens, 'to'):
                        import torch
                        greedy_tokens = greedy_tokens.to(dtype=torch.int32)
                else:
                    greedy_tokens = logits
                return (*values, greedy_tokens)
            finally:
                owner._inside_step_graph = False

        programs[key] = (ht.hpu.wrap_in_hpu_graph_func(
            spec_tensor_program, disable_tensor_cache=False, asynchronous=False), sampling)
    program, sampling = programs[key]
    values = program(token_ids, position_ids, metadata, logits_indices)
    runner._step_graph_spec_sample = (
        ((runner.batch_changed, tuple(batch.req_ids[:batch.num_reqs])), values[4])
        if sampling is None else
        ((runner.batch_changed, tuple(batch.req_ids[:batch.num_reqs])), values[4], sampling))
    runner._step_graph_calls = getattr(runner, '_step_graph_calls', 0) + 1
    return values[:4]


def _uniform_greedy_verify(greedy_i32, draft_i32, batch_size, max_spec_len):
    import torch
    greedy = greedy_i32.view(batch_size, max_spec_len + 1)
    draft = draft_i32.view(batch_size, max_spec_len)
    matches = (greedy[:, :max_spec_len] == draft)
    accepted_before = matches.to(dtype=torch.int32).cumprod(dim=1)
    first = greedy[:, :1]
    rest = torch.where(accepted_before.to(dtype=torch.bool), greedy[:, 1:], -1)
    return torch.cat((first, rest), dim=1)


def _verify_greedy_speculative(greedy_tokens, spec_metadata, runner=None):
    import torch

    draft_token_ids = getattr(spec_metadata, '_cpu_draft_token_ids', None)
    if draft_token_ids is None:
        draft_token_ids = spec_metadata.draft_token_ids
    cu_num_draft_tokens = getattr(spec_metadata, 'cu_num_draft_tokens', None)
    num_draft_tokens = getattr(spec_metadata, 'num_draft_tokens', None)
    if num_draft_tokens is not None:
        draft_counts = [int(n) for n in num_draft_tokens]
        batch_size = len(draft_counts)
    elif cu_num_draft_tokens is not None:
        cu_list = [int(x) for x in cu_num_draft_tokens.detach().cpu().reshape(-1).tolist()]
        batch_size = len(cu_list)
        draft_counts = ([cu_list[0]] + [cu_list[i] - cu_list[i - 1] for i in range(1, batch_size)]
                        if batch_size else [])
    else:
        raise ValueError('spec_metadata must provide num_draft_tokens or cu_num_draft_tokens')

    max_spec_len = int(getattr(spec_metadata, 'max_spec_len',
                               max(draft_counts) if draft_counts else 0))
    # RejectionSampler.parse_output immediately calls output_token_ids.cpu().numpy().
    # Copying greedy_tokens (B*(K+1) int32s) to CPU once and verifying against CPU
    # draft_token_ids eliminates an HPU graph launch plus a second D2H sync.
    device = torch.device('cpu')
    greedy_i32 = (greedy_tokens if (greedy_tokens.device.type == 'cpu' and greedy_tokens.dtype == torch.int32)
                  else greedy_tokens.to(device=device, dtype=torch.int32)).reshape(-1)
    draft_i32 = (draft_token_ids if (draft_token_ids.device.type == 'cpu' and draft_token_ids.dtype == torch.int32)
                 else draft_token_ids.to(device=device, dtype=torch.int32)).reshape(-1)

    if batch_size == 0:
        return torch.empty((0, max_spec_len + 1), dtype=torch.int32, device=device)

    if max_spec_len == 0:
        bonus_indices = getattr(spec_metadata, 'bonus_logits_indices', None)
        if bonus_indices is not None and bonus_indices.numel() == batch_size:
            return greedy_i32[bonus_indices.to(device=device, dtype=torch.long)].reshape(batch_size, 1)
        return greedy_i32[:batch_size].reshape(batch_size, 1)

    if (greedy_i32.numel() == batch_size * (max_spec_len + 1) and
            draft_i32.numel() == batch_size * max_spec_len and
            all(n == max_spec_len for n in draft_counts)):
        return _uniform_greedy_verify(greedy_i32, draft_i32, batch_size, max_spec_len)

    total_draft = sum(draft_counts)
    if draft_i32.numel() == batch_size * max_spec_len:
        draft = draft_i32.view(batch_size, max_spec_len)
    else:
        draft = torch.full((batch_size, max_spec_len), -1, dtype=torch.int32, device=device)
        cursor = 0
        for i, n in enumerate(draft_counts):
            if n > 0:
                draft[i, :n] = draft_i32[cursor:cursor + n]
                cursor += n

    target_indices = getattr(spec_metadata, 'target_logits_indices', None)
    bonus_indices = getattr(spec_metadata, 'bonus_logits_indices', None)
    has_valid_bonus = (
        bonus_indices is not None and bonus_indices.numel() == batch_size and
        not (greedy_i32.numel() > 1 and int(bonus_indices.detach().cpu().max().item()) == 0))
    if (has_valid_bonus and target_indices is not None and
            target_indices.numel() == total_draft):
        greedy = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32, device=device)
        target_flat = greedy_i32[target_indices.to(device=device, dtype=torch.long)]
        bonus_flat = greedy_i32[bonus_indices.to(device=device, dtype=torch.long)]
        cursor = 0
        for i, n in enumerate(draft_counts):
            if n > 0:
                greedy[i, :n] = target_flat[cursor:cursor + n]
                cursor += n
            greedy[i, n] = bonus_flat[i]
    elif greedy_i32.numel() == batch_size * (max_spec_len + 1):
        greedy = greedy_i32.view(batch_size, max_spec_len + 1).clone()
        if has_valid_bonus:
            bonus_flat = greedy_i32[bonus_indices.to(device=device, dtype=torch.long)]
            for i, n in enumerate(draft_counts):
                greedy[i, n] = bonus_flat[i]
    else:
        greedy = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32, device=device)
        cursor = 0
        for i, n in enumerate(draft_counts):
            greedy[i, :n + 1] = greedy_i32[cursor:cursor + n + 1]
            cursor += n + 1

    counts = torch.tensor(draft_counts, dtype=torch.int32, device=device).unsqueeze(1)
    pos = torch.arange(max_spec_len, dtype=torch.int32, device=device).unsqueeze(0)
    matches = (greedy[:, :max_spec_len] == draft) & (pos < counts)
    accepted_before = matches.to(dtype=torch.int32).cumprod(dim=1)
    first = greedy[:, :1]
    rest = torch.where(accepted_before.to(dtype=torch.bool), greedy[:, 1:], -1)
    return torch.cat((first, rest), dim=1)


def sample_speculative(runner, fallback_sample_decode, batch_changed,
                       logits_device, request_ids, spec_metadata):
    from vllm.v1.outputs import SamplerOutput

    cached = getattr(runner, '_step_graph_spec_sample', None)
    runner._step_graph_spec_sample = None
    if (spec_metadata is None or cached is None or request_ids is None or
            cached[0] != (batch_changed, tuple(request_ids))):
        return fallback_sample_decode(batch_changed, logits_device, request_ids, spec_metadata)

    update_async = getattr(getattr(runner, 'input_batch', None), 'update_async_output_token_ids', None)
    if callable(update_async):
        update_async()
    sampling_metadata = cached[2] if (len(cached) >= 3 and cached[2] is not None and not batch_changed) else None
    if sampling_metadata is None:
        try:
            pad_to = logits_device.shape[0] if hasattr(logits_device, 'shape') else None
            sampling_metadata = runner._prepare_sampling(batch_changed, request_ids, pad_to)
        except TypeError:
            sampling_metadata = runner._prepare_sampling(batch_changed, request_ids)
    if (not getattr(sampling_metadata, 'all_greedy', False) or
            not getattr(sampling_metadata, 'no_penalties', False) or
            getattr(sampling_metadata, 'max_num_logprobs', None) is not None):
        return fallback_sample_decode(batch_changed, logits_device, request_ids, spec_metadata)

    output_token_ids = _verify_greedy_speculative(cached[1], spec_metadata, runner=runner)
    return SamplerOutput(sampled_token_ids=output_token_ids, logprobs_tensors=None), sampling_metadata


def configure(worker, enabled):
    """Idle diagnostic switch for same-resident-model A/B, never a hot-path RPC."""
    import torch
    import habana_frameworks.torch.core as hc
    if type(enabled) is not bool:
        raise ValueError('enabled must be bool')
    runner = worker.model_runner
    if getattr(runner, 'scheduler_output', None) is not None:
        raise RuntimeError('Cannot switch graph scope during a scheduled step')
    hc.mark_step()
    torch.hpu.synchronize()
    runner._step_graph_override = enabled
    runner._step_graph_sample = None
    runner._step_graph_spec_sample = None
    return dict(rank=worker.rank, enabled=enabled,
                calls=getattr(runner, '_step_graph_calls', 0))
