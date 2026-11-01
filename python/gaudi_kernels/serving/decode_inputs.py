"""Reuse validated decode input layouts; values still come from the scheduler.

This is a vLLM host adapter, not a second device executor. Shape/page/layout
changes rebuild through the original runner. Uploads remain ordinary copy_.
"""
import numpy as np


def eligibility_reason(runner, count, scheduled):
    if count <= 0 or getattr(runner, '_step_decode_count', 0) != count:
        return 'not synchronous decode'
    parallel = runner.vllm_config.parallel_config
    if (parallel.data_parallel_size != 1 or parallel.pipeline_parallel_size != 1 or
            getattr(runner.vllm_config, 'max_concurrent_batches', 1) != 1):
        return 'concurrent or padded parallel layout'
    if getattr(getattr(runner, '_input_audit', None), 'audit_inputs', None) is not None:
        return 'input audit requires every original transfer'
    for name in ('uses_mrope', 'model_has_chunked_attention', 'num_mamba_like_layers',
                 'use_async_scheduling'):
        if getattr(runner, name, None):
            return name
    spec = getattr(runner, 'speculative_config', None)
    if spec is not None and getattr(spec, 'num_speculative_tokens', 0) <= 0:
        return 'speculative_config'
    defrag = getattr(runner, 'defragmenter', None)
    # The plugin always constructs this object. Its disabled resolve() is the
    # identity; an enabled or unknown mapper still uses the original builder.
    if defrag is not None and getattr(defrag, 'enabled', True):
        return 'active KV page defragmenter'
    if len(scheduled) < count or any(n < 1 for n in scheduled[:count]):
        return 'multiple queries per request'
    if spec is None and any(n != 1 for n in scheduled[:count]):
        return 'multiple queries per request'
    return None


def eligible(runner, count, scheduled):
    return eligibility_reason(runner, count, scheduled) is None


def prepare_single_query_values(runner, scheduler, count):
    """Populate the original CPU staging buffers for one or more queries per request.

    The general builder expands ragged row indices and computes a cumulative
    sum. For a validated decode layout we slice actual scheduler positions and
    tokens directly into the staging buffers, including after speculative rollback.
    """
    ids = runner.input_batch.req_ids
    if len(ids) != count:
        return None
    scheduled = [int(scheduler.num_scheduled_tokens[rid]) for rid in ids]
    total = sum(scheduled)
    if scheduler.total_num_scheduled_tokens != total or not eligible(runner, count, scheduled):
        return None
    positions = runner.input_batch.num_computed_tokens_cpu[:count]
    tokens = runner.input_batch.token_ids_cpu_tensor.numpy()
    destination = runner.input_ids_cpu.numpy()
    if total == count:
        if (tokens.dtype != destination.dtype or len(destination) < count or
                len(runner.positions_np) < count or np.any(positions < 0) or
                np.any(positions >= tokens.shape[1])):
            return None
        runner.positions_np[:count] = positions
        destination[:count] = tokens[runner.arange_np[:count], positions]
        return scheduled
    counts = np.asarray(scheduled, dtype=np.int64)
    ends = positions + counts
    if (tokens.dtype != destination.dtype or len(destination) < total or
            len(runner.positions_np) < total or len(runner.arange_np) < max(scheduled) or
            np.any(positions < 0) or np.any(ends > tokens.shape[1])):
        return None
    if count == 1:
        start = int(positions[0])
        runner.positions_np[:total] = start + runner.arange_np[:total]
        destination[:total] = tokens[0, start:start + total]
        return scheduled
    cursor = 0
    for i, n in enumerate(scheduled):
        start = int(positions[i])
        next_cursor = cursor + n
        runner.positions_np[cursor:next_cursor] = start + runner.arange_np[:n]
        destination[cursor:next_cursor] = tokens[i, start:start + n]
        cursor = next_cursor
    return scheduled


def _draft_counts(runner, count, scheduler_output):
    if scheduler_output is None:
        return ()
    spec_tokens = getattr(scheduler_output, 'scheduled_spec_decode_tokens', None)
    if not spec_tokens:
        return ()
    req_ids = runner.input_batch.req_ids[:count]
    req_id_to_index = getattr(runner.input_batch, 'req_id_to_index', None)
    if req_id_to_index is not None:
        for i, rid in enumerate(req_ids):
            if req_id_to_index.get(rid, i) != i:
                return None
    if sum(1 for rid in req_ids if rid in spec_tokens) != len(spec_tokens):
        return None
    return tuple(len(spec_tokens.get(rid, ())) for rid in req_ids)


def _draft_token_rows(num_draft_tokens, max_num_sampled_tokens):
    rows = []
    for batch_id, n_draft in enumerate(num_draft_tokens):
        base = batch_id * max_num_sampled_tokens
        rows.extend(base + i + 1 for i in range(n_draft))
        if n_draft + 1 < max_num_sampled_tokens:
            rows.extend([0] * (max_num_sampled_tokens - n_draft - 1))
    return np.asarray(rows, dtype=np.int64)


def prepare_spec_decode_inputs(runner, original, scheduler_output, logits_indices,
                               token_ids_device, max_num_sampled_tokens):
    import torch
    if scheduler_output is None or not getattr(scheduler_output, 'scheduled_spec_decode_tokens', None):
        return logits_indices, None
    count = getattr(runner, '_step_decode_count', 0) or len(runner.input_batch.req_ids)
    draft_counts = _draft_counts(runner, count, scheduler_output)
    if draft_counts is None:
        return original(scheduler_output, logits_indices, token_ids_device, max_num_sampled_tokens)
    key = (draft_counts, int(logits_indices.numel()), int(max_num_sampled_tokens), str(runner.device))
    capture = getattr(runner, '_prepared_input_capture', None)
    token_ids_cpu = capture['token_ids'][0] if capture is not None and 'token_ids' in capture else None
    cached = getattr(runner, '_prepared_spec_decode_inputs', None)
    if cached is not None and cached[0] == key:
        _, cached_logits_indices, spec_decode_metadata, draft_token_rows, draft_tokens_source, draft_token_indices = cached
        if token_ids_cpu is not None:
            draft_tokens_source.reshape(-1, 1).numpy()[:, 0] = token_ids_cpu.reshape(-1, 1).numpy()[draft_token_rows, 0]
            spec_decode_metadata.draft_token_ids.copy_(draft_tokens_source, non_blocking=True)
            spec_decode_metadata._cpu_draft_token_ids = draft_tokens_source
        else:
            if draft_token_indices is None:
                draft_token_indices = spec_decode_metadata.target_logits_indices + 1
            spec_decode_metadata.draft_token_ids.copy_(token_ids_device[draft_token_indices])
            spec_decode_metadata._cpu_draft_token_ids = None
        runner._prepared_spec_decode_input_hits = getattr(runner, '_prepared_spec_decode_input_hits', 0) + 1
        return cached_logits_indices, spec_decode_metadata
    out_logits_indices, spec_decode_metadata = original(
        scheduler_output, logits_indices, token_ids_device, max_num_sampled_tokens)
    if spec_decode_metadata is not None:
        draft_token_rows = _draft_token_rows(spec_decode_metadata.num_draft_tokens, int(max_num_sampled_tokens))
        draft_tokens_source = torch.empty(
            spec_decode_metadata.draft_token_ids.shape,
            dtype=spec_decode_metadata.draft_token_ids.dtype,
            device='cpu')
        draft_token_indices = None
        if token_ids_cpu is not None:
            draft_tokens_source.reshape(-1, 1).numpy()[:, 0] = token_ids_cpu.reshape(-1, 1).numpy()[draft_token_rows, 0]
            spec_decode_metadata.draft_token_ids = draft_tokens_source.to(runner.device, non_blocking=True)
            spec_decode_metadata._cpu_draft_token_ids = draft_tokens_source
        else:
            draft_token_indices = spec_decode_metadata.target_logits_indices + 1
            spec_decode_metadata._cpu_draft_token_ids = None
        runner._prepared_spec_decode_inputs = (
            key, out_logits_indices, spec_decode_metadata,
            draft_token_rows, draft_tokens_source, draft_token_indices)
    return out_logits_indices, spec_decode_metadata


class PreparedDecodeInputs:
    def __init__(self, runner, count, contexts, table, result, transfers,
                 scheduled=None, scheduler_output=None):
        import torch
        self.count = count
        self.scheduled = (tuple(int(n) for n in scheduled[:count])
                          if scheduled is not None else (1,) * count)
        self.num_tokens = max(self.scheduled)
        self.total_tokens = sum(self.scheduled)
        self.contiguous_rows = all(n == self.num_tokens for n in self.scheduled)
        self.block_size = runner.attn_block_size
        self.window = runner.sliding_window if runner.interleaved_sliding_window else None
        self.request_ids = tuple(runner.input_batch.req_ids[:count])
        self.layout = (runner._PAD_BLOCK_ID, runner.use_contiguous_pa, runner.model_config.dtype,
                       id(runner.bucketing_manager))
        self.draft_counts = _draft_counts(runner, count, scheduler_output)
        self.active_rows = np.asarray(
            [i * self.num_tokens + j for i, n in enumerate(self.scheduled) for j in range(n)],
            dtype=np.int64)
        self.req_for_query = np.asarray(
            [i for i, n in enumerate(self.scheduled) for j in range(n)],
            dtype=np.int64)
        self.offset_for_query = np.asarray(
            [j for n in self.scheduled for j in range(n)],
            dtype=np.int64)
        base_contexts = np.asarray(contexts[:count], dtype=np.int64)
        query_positions = self._query_positions(base_contexts)
        self.query_page_counts = query_positions // self.block_size + 1
        req_ends = base_contexts + np.asarray(self.scheduled, dtype=np.int64) - 1
        self.page_counts = req_ends // self.block_size + 1
        self.pages = tuple(table[i, :int(n)].numpy().copy() for i, n in enumerate(self.page_counts))
        self.window_starts = self._window_starts(query_positions)
        self.result = result
        self.transfers = transfers
        roles = ('positions', 'block_usage', 'slot_mapping', 'token_ids')
        if self.window is not None:
            roles += ('window_block_usage',)
        upload_sources = [transfers[role][0] for role in roles]
        upload_destinations = [transfers[role][1] for role in roles]
        expected = {'positions', 'token_ids', 'slot_mapping', 'logits_indices',
                    'block_list', 'block_groups', 'block_usage'}
        if self.window is not None:
            expected |= {'window_block_list', 'window_block_groups', 'window_block_usage'}
        if transfers.keys() != expected:
            raise ValueError('Unexpected decode input transfer contract')
        self.arrays = {name: transfers[name][0].numpy() for name in
                       ('positions', 'token_ids', 'slot_mapping')}
        self.usage = []
        last_pages_per_query = np.asarray(
            [self.pages[int(req_idx)][int(page_count) - 1]
             for req_idx, page_count in zip(self.req_for_query, self.query_page_counts)],
            dtype=np.int64)
        self.slot_bases = last_pages_per_query * self.block_size
        total_groups = count * self.num_tokens
        last_pages_by_row = np.full(total_groups, -1, dtype=np.int64)
        last_pages_by_row[self.active_rows] = last_pages_per_query
        row_to_query = np.full(total_groups, -1, dtype=np.int64)
        row_to_query[self.active_rows] = np.arange(self.total_tokens, dtype=np.int64)
        for prefix in ('', 'window_') if self.window is not None else ('',):
            ids = transfers[prefix+'block_list'][0].numpy()
            groups = transfers[prefix+'block_groups'][0].numpy()
            # Contiguous PA can leave only the final owner of a shared page.
            # Use the original builder's group/physical-page mapping verbatim.
            active = (groups >= 0) & (groups < total_groups)
            slots = np.flatnonzero(active)
            slots = slots[ids[slots] == last_pages_by_row[groups[slots]]]
            source = transfers[prefix+'block_usage'][0]
            scratch = source.to(dtype=torch.float32, copy=True)
            self.usage.append((prefix+'block_usage', source, scratch, scratch.numpy(),
                               slots, row_to_query[groups[slots]]))
        spec_meta = getattr(result, 'spec_decode_metadata', None)
        cached_spec = getattr(runner, '_prepared_spec_decode_inputs', None)
        if spec_meta is None:
            self.draft_token_rows = None
            self.draft_tokens_array = None
        else:
            if cached_spec is not None and cached_spec[2] is spec_meta:
                self.draft_token_rows = cached_spec[3]
                draft_source = cached_spec[4]
            else:
                self.draft_token_rows = _draft_token_rows(spec_meta.num_draft_tokens, self.num_tokens)
                draft_source = torch.empty(
                    spec_meta.draft_token_ids.shape,
                    dtype=spec_meta.draft_token_ids.dtype,
                    device='cpu')
            spec_meta._cpu_draft_token_ids = draft_source
            self.draft_tokens_array = draft_source.reshape(-1, 1).numpy()
            self.draft_tokens_array[:, 0] = self.arrays['token_ids'][self.draft_token_rows, 0]
            upload_sources.append(draft_source)
            upload_destinations.append(spec_meta.draft_token_ids)
        self.upload_sources = tuple(upload_sources)
        self.upload_destinations = tuple(upload_destinations)

    def _query_positions(self, base_contexts):
        if self.num_tokens == 1:
            return base_contexts
        if self.count == 1:
            return int(base_contexts[0]) + self.offset_for_query
        return base_contexts[self.req_for_query] + self.offset_for_query

    def _window_starts(self, positions):
        if self.window is None:
            return None
        return np.maximum(0, (np.asarray(positions)-self.window+1)//self.block_size)

    def update(self, runner, count, contexts, table, scheduled=None, scheduler_output=None):
        current_scheduled = (tuple(int(n) for n in scheduled[:count])
                             if scheduled is not None else (1,) * count)
        if (count != self.count or current_scheduled != self.scheduled or
                self.block_size != runner.attn_block_size or
                self.layout != (runner._PAD_BLOCK_ID, runner.use_contiguous_pa, runner.model_config.dtype,
                                id(runner.bucketing_manager)) or
                self.window != (runner.sliding_window if runner.interleaved_sliding_window else None) or
                self.request_ids != tuple(runner.input_batch.req_ids[:count]) or
                self.draft_counts is None or
                _draft_counts(runner, count, scheduler_output) != self.draft_counts):
            return None
        base_contexts = np.asarray(contexts[:count], dtype=np.int64)
        query_positions = self._query_positions(base_contexts)
        if (not np.array_equal(query_positions // self.block_size + 1, self.query_page_counts) or
                not np.array_equal(self._window_starts(query_positions), self.window_starts)):
            return None
        # Compare actual page IDs, not a guessed "same request means same KV"
        # invariant. In-place page-table changes invalidate before any upload.
        page_table = table.numpy()
        for i, pages in enumerate(self.pages):
            if not np.array_equal(page_table[i, :len(pages)], pages):
                return None
        actual_positions = runner.positions_cpu.numpy()[:self.total_tokens]
        if not np.array_equal(actual_positions, query_positions):
            return None
        rows = slice(None, self.total_tokens) if self.contiguous_rows else self.active_rows
        self.arrays['positions'][rows, 0] = actual_positions
        self.arrays['token_ids'][rows, 0] = runner.input_ids_cpu.numpy()[:self.total_tokens]
        offsets = query_positions % self.block_size
        usage = offsets + 1
        self.arrays['slot_mapping'][rows, 0] = self.slot_bases + offsets
        if self.draft_token_rows is not None:
            self.draft_tokens_array[:, 0] = self.arrays['token_ids'][self.draft_token_rows, 0]
            runner._prepared_spec_decode_input_hits = getattr(runner, '_prepared_spec_decode_input_hits', 0) + 1
        for role, source, scratch, values, slots, query_indices in self.usage:
            values[slots] = usage[query_indices]
            source.copy_(scratch)  # Preserve PyTorch's original dtype rounding.
        # Immutable page lists, groups and logits indices keep their original
        # HPU contents. Only scheduler values that changed are uploaded.
        import torch
        # Standard Torch operator: the bridge batches eligible independent H2D
        # destinations and preserves the original copy path for everything else.
        torch._foreach_copy_(self.upload_destinations,self.upload_sources,non_blocking=True)
        runner._prepared_decode_input_hits = getattr(runner, '_prepared_decode_input_hits', 0) + 1
        return self.result


def configure(worker, enabled):
    import torch
    import habana_frameworks.torch.core as hc
    runner = worker.model_runner
    if type(enabled) is not bool or getattr(runner, 'scheduler_output', None) is not None:
        raise ValueError('Input layout policy can change only between scheduled steps')
    hc.mark_step()
    torch.hpu.synchronize()
    runner._prepared_decode_inputs_enabled = enabled
    runner._prepared_decode_inputs = None
    runner._prepared_spec_decode_inputs = None
    runner._prepared_decode_input_hits = 0
    runner._prepared_spec_decode_input_hits = 0
    return dict(rank=worker.rank, enabled=enabled)
