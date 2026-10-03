"""Encode actual vLLM CPU scheduling state without importing vLLM or HPU.

This adapter describes work; it does not admit native speculation, alter the
scheduler, commit KV, or replace the legacy runner's attention metadata.
"""
from gaudi_kernels.engine.token_batch import RequestTokens, TokenBatch


def from_vllm(input_batch, scheduler_output, requests, *, capacity=None):
    ids = tuple(input_batch.req_ids[:input_batch.num_reqs])
    counts = scheduler_output.num_scheduled_tokens
    if set(counts) != set(ids):
        raise ValueError('Scheduled requests differ from the current input batch')
    computed = input_batch.num_computed_tokens_cpu
    prompts = input_batch.num_prompt_tokens
    storage = input_batch.token_ids_cpu
    spec = scheduler_output.scheduled_spec_decode_tokens
    encoded = []
    for index, rid in enumerate(ids):
        count = counts[rid]
        if type(count) is not int or count < 1:
            raise ValueError('Invalid scheduled token count')
        start = int(computed[index])
        if start < 0:
            raise ValueError('Negative request cursor')
        row = storage[index][start:start + count]
        values = row.tolist() if hasattr(row, 'tolist') else list(row)
        if len(values) != count or any(type(token) is not int for token in values):
            raise ValueError('Scheduled tokens exceed available CPU storage or index type')
        drafts = spec.get(rid, ())
        if drafts:
            if count != len(drafts) + 1:
                raise ValueError('Verify input must contain one anchor and all scheduled drafts')
            if values[1:] != list(drafts):
                raise ValueError('Scheduled draft IDs differ from actual input token ownership')
            kind, outputs = 'verify', tuple(range(count))
        else:
            prompt_length = int(prompts[index])
            # Catch-up of previously generated tokens can be multirow too.
            kind = 'prefill' if start < prompt_length or count > 1 else 'decode'
            known = requests[rid].num_tokens
            if start + count > known:
                raise ValueError('Scheduled non-speculative input exceeds known request tokens')
            outputs = (count - 1,) if start + count == known else ()
        encoded.append(RequestTokens(rid, tuple(values), start, kind, outputs))
    batch = TokenBatch(tuple(encoded), capacity)
    if batch.num_tokens != scheduler_output.total_num_scheduled_tokens:
        raise ValueError('Scheduled total token count differs from request counts')
    return batch
