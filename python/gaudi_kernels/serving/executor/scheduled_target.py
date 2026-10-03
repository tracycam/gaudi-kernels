"""Experimental compact non-speculative consumer of vLLM scheduling and KV.

The allocator and sampler remain vLLM-owned. This correctness path synchronizes
and uses tiled FP32 attention; it is not native replay/performance admission.
"""
from types import SimpleNamespace

import torch

from gaudi_kernels.serving.executor.packed_attention import PackedAttentionMetadata


def token_slots(page_table, length, block_size, *, resolve=lambda block: block):
    if type(length) is not int or length < 1 or type(block_size) is not int or block_size < 1:
        raise ValueError('Invalid token/page geometry')
    pages = (length + block_size - 1) // block_size
    if len(page_table) < pages:
        raise ValueError('Incomplete scheduled page table')
    blocks = [int(resolve(int(block))) for block in page_table[:pages]]
    if any(block < 0 for block in blocks) or len(set(blocks)) != pages:
        raise ValueError('Invalid or aliased logical pages')
    return tuple(blocks[position // block_size] * block_size + position % block_size
                 for position in range(length))


class ScheduledKVBinding:
    """Borrow allocator-owned ordinary KV; never admit candidate verification."""
    def __init__(self, runner):
        self.runner = runner
        self.pending = None

    def prepare(self, batch):
        if self.pending is not None or any(request.kind == 'verify' for request in batch.requests):
            raise ValueError('Scheduled KV requires non-speculative work and no pending step')
        runner = self.runner
        layers = runner.vllm_config.compilation_config.static_forward_context
        result = {}
        for group_index, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            block_size = group.kv_cache_spec.block_size
            table = runner.input_batch.block_table[group_index].get_cpu_tensor()
            visible, candidates, key_positions, query_positions, all_slots = [], [], [], [], []
            past_slots, writes = set(), []
            for request in batch.requests:
                index = runner.input_batch.req_id_to_index[request.request_id]
                end = request.start_position + request.query_length
                slots = token_slots(table[index].tolist(), end, block_size, resolve=runner._resolve_block)
                all_slots.extend(slots)
                past_slots.update(slots[:request.start_position])
                candidate = slots[request.start_position:]
                writes.extend(candidate)
                candidates.append(candidate)
                visible.append(torch.tensor(slots, device=runner.device, dtype=torch.int64))
                key_positions.append(torch.arange(end, device=runner.device))
                query_positions.append(torch.arange(request.start_position, end, device=runner.device))
            if len(set(writes)) != len(writes) or set(writes) & past_slots:
                raise ValueError('Scheduled writes alias live KV or another request')
            slot_mapping = torch.tensor(writes, device=runner.device, dtype=torch.int64)
            for name in group.layer_names:
                layer = layers[name]
                cache = layer.kv_cache
                if (not isinstance(cache, tuple) or len(cache) < 2 or
                        any(value.dtype != torch.bfloat16 or value.ndim != 3 for value in cache[:2]) or
                        any(slot >= cache[0].shape[0] or slot >= cache[1].shape[0] for slot in all_slots)):
                    raise ValueError('Unsupported allocator KV layout or physical slot')
                owner = SimpleNamespace(caches={name: cache[:2]}, pending=None)
                metadata = PackedAttentionMetadata(batch, owner, tuple(candidates), slot_mapping,
                    tuple(visible), tuple(key_positions), tuple(query_positions))
                owner.pending = metadata
                result[name] = metadata
        if not result:
            raise ValueError('No allocator-owned attention layers')
        self.pending = result
        return result

    def commit(self, metadata, prefix_lengths):
        if metadata is not self.pending:
            raise ValueError('Stale scheduled KV step')
        batch = next(iter(metadata.values())).batch
        if tuple(prefix_lengths) != batch.query_lengths:
            raise ValueError('Speculative KV prefix commits require a separate allocator lease')
        for value in metadata.values():
            value.session.pending = None
        self.pending = None

    def abort(self, metadata):
        if metadata is not self.pending:
            raise ValueError('Stale scheduled KV step')
        # Candidate slots were outside all visible past; partial writes remain
        # unreachable. Caller must fail the step, never fabricate model output.
        for value in metadata.values():
            value.session.pending = None
        self.pending = None


def sample_scheduled(runner, grammar_output):
    from vllm.v1.outputs import ModelRunnerOutput
    from gaudi_kernels.serving.executor.scheduled_tokens import from_vllm
    from gaudi_kernels.serving.executor.packed_target import PackedTargetExecutor
    from gaudi_kernels.engine.context import context
    if grammar_output is not None or getattr(runner, '_packed_scheduled_failed', False):
        raise ValueError('Packed scheduled correctness path has unsupported grammar or failed state')
    scheduler = runner.scheduler_output
    if scheduler is None:
        return None
    if runner.warmup_mode:
        raise ValueError('Packed scheduling is enabled only after ordinary initialization')
    batch = from_vllm(runner.input_batch, scheduler, runner.requests)
    for request in batch.requests:
        params = runner.requests[request.request_id].sampling_params
        if (params is None or params.temperature != 0 or params.logprobs is not None or
                params.prompt_logprobs is not None):
            raise ValueError('Packed scheduled probe admits greedy text without logprob requests')
        if request.kind == 'verify':
            raise ValueError('Speculation requires protected candidate KV')
    runner.scheduler_output = None
    runner.warmup_mode = False
    output = [[] for _ in range(runner.input_batch.num_reqs)]
    try:
        executor = PackedTargetExecutor(runner, session=ScheduledKVBinding(runner))
        target = executor.execute(batch)
        owners = [rid for rid, _ in target.output_owners]
        if target.logits is not None:
            sampled, _ = runner._run_sampling(True, target.logits, list(batch.request_ids),
                                              len(owners), owners)
            ids = runner.copy_output_to_cpu('sampled_tokens', sampled.sampled_token_ids).reshape(-1).tolist()
            if len(ids) != len(owners):
                raise ValueError('Sampler output ownership mismatch')
            for rid, token in zip(owners, ids):
                output[runner.input_batch.req_id_to_index[rid]] = [token]
        executor.commit(target, batch.query_lengths)
        for index, rid in enumerate(runner.input_batch.req_ids[:runner.input_batch.num_reqs]):
            tokens = output[index]
            start = int(runner.input_batch.num_tokens_no_spec[index])
            end = start + len(tokens)
            if end > runner.max_model_len + 1:
                raise ValueError('Sampled output exceeds request storage')
            runner.input_batch.token_ids_cpu[index, start:end] = tokens
            runner.input_batch.num_tokens_no_spec[index] = end
            runner.input_batch.num_tokens[index] = end
            runner.requests[rid].output_token_ids.extend(tokens)
        records = runner._packed_scheduled_records
        if len(records) >= 512:
            raise ValueError('Packed scheduled diagnostic capacity exhausted')
        record = {'request_ids': batch.request_ids, 'query_lengths': batch.query_lengths,
                  'valid_rows': batch.num_tokens, 'kinds': [r.kind for r in batch.requests],
                  'positions': batch.encoded()['positions'], 'input_ids': batch.encoded()['token_ids'],
                  'logits_indices': batch.logits_indices,
                  'output_owners': target.output_owners, 'model_calls': 1}
        if runner.is_driver_worker and target.logits is not None:
            path = context().startup.run_dir/'packed-scheduled'
            path.mkdir(exist_ok=True)
            destination = path/f'{len(records)}.pt'
            torch.save({'logits': target.logits.cpu(), 'input_ids': batch.encoded()['token_ids'],
                        'positions': batch.encoded()['positions']}, destination)
            record['path'] = str(destination)
        records.append(record)
        return ModelRunnerOutput(req_ids=list(batch.request_ids),
            req_id_to_index=dict(runner.input_batch.req_id_to_index), sampled_token_ids=output,
            logprobs=None, prompt_logprobs_dict={}, pooler_output=[], kv_connector_output=None)
    except BaseException:
        runner._packed_scheduled_failed = True
        raise
