# SPDX-License-Identifier: Apache-2.0
"""Shard-local output reduction and OutputSpec execution contracts."""
from __future__ import annotations

import torch

from gaudi_kernels.engine.token_batch import OutputSpec

try:
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        tensor_model_parallel_all_gather,
    )
except ImportError:  # pragma: no cover - standalone unit tests without vllm
    get_tensor_model_parallel_rank = None
    tensor_model_parallel_all_gather = None


def _tp_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    fn = tensor_model_parallel_all_gather
    if fn is None:
        from vllm.distributed import tensor_model_parallel_all_gather as fn
    try:
        return fn(tensor, dim=dim)
    except TypeError:
        return fn(tensor)


def _tp_rank_for_head(lm_head, tp_size: int) -> int:
    tp_rank = getattr(lm_head, 'tp_rank', None)
    if tp_rank is not None:
        return int(tp_rank)
    if tp_size > 1:
        fn = get_tensor_model_parallel_rank
        if fn is None:
            from vllm.distributed import get_tensor_model_parallel_rank as fn
        return int(fn())
    return 0


def _resolve_lm_head_and_processor(model):
    curr = model
    visited = set()
    while curr is not None and id(curr) not in visited:
        visited.add(id(curr))
        lm_head = getattr(curr, 'lm_head', None)
        logits_processor = getattr(curr, 'logits_processor', None)
        if lm_head is not None and logits_processor is not None:
            return curr, lm_head, logits_processor
        if hasattr(curr, 'language_model'):
            curr = getattr(curr, 'language_model')
        elif hasattr(curr, 'model'):
            curr = getattr(curr, 'model')
        else:
            break
    return model, getattr(model, 'lm_head', None), getattr(model, 'logits_processor', None)


def _supports_shard_local_reduction(lm_head, logits_processor) -> bool:
    if lm_head is None or logits_processor is None:
        return False
    if getattr(logits_processor, 'logits_as_input', False):
        return False
    if int(getattr(lm_head, 'num_added_embeddings', 0) or 0) != 0:
        return False
    quant_method = getattr(lm_head, 'quant_method', None)
    if quant_method is None or not callable(getattr(quant_method, 'apply', None)):
        return False
    tp_size = int(getattr(lm_head, 'tp_size', 1) or 1)
    return bool(getattr(logits_processor, 'use_all_gather', False) or tp_size > 1)


def _compute_shard_logits(
    lm_head,
    logits_processor,
    hidden_states: torch.Tensor,
    embedding_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int, int]:
    bias = embedding_bias if embedding_bias is not None else getattr(lm_head, 'bias', None)
    head_dtype = getattr(logits_processor, 'head_dtype', None)
    if (
        head_dtype is not None
        and head_dtype != hidden_states.dtype
        and callable(getattr(logits_processor, '_apply_head', None))
    ):
        logits = logits_processor._apply_head(lm_head, hidden_states, bias)
    else:
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=bias)

    soft_cap = getattr(logits_processor, 'soft_cap', None)
    if soft_cap is not None:
        logits = torch.tanh(logits / soft_cap) * soft_cap

    scale = getattr(logits_processor, 'scale', 1.0)
    if scale is not None and scale != 1.0:
        logits = logits * scale

    tp_size = int(getattr(lm_head, 'tp_size', 1) or 1)
    tp_rank = _tp_rank_for_head(lm_head, tp_size)
    partition_size = int(getattr(lm_head, 'num_embeddings_per_partition', logits.shape[-1]))

    # Out-of-place slice truncation avoids in-place slice mutation inside HPU Graph.
    if tp_rank == tp_size - 1:
        num_padded = getattr(lm_head, 'num_embeddings_padded', None)
        org_vocab = getattr(logits_processor, 'org_vocab_size', getattr(lm_head, 'org_vocab_size', None))
        if num_padded is not None and org_vocab is not None:
            num_pad = int(num_padded) - int(org_vocab)
        else:
            shard_indices = getattr(lm_head, 'shard_indices', None)
            num_pad = int(getattr(shard_indices, 'num_org_vocab_padding', 0) or 0)
        if 0 < num_pad < logits.shape[-1]:
            logits = logits[..., :-num_pad]

    offset = tp_rank * partition_size
    return logits, tp_rank, tp_size, offset


def _compute_full_logits(
    model,
    hidden_states: torch.Tensor,
    embedding_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if embedding_bias is not None:
        try:
            return model.compute_logits(hidden_states, embedding_bias)
        except TypeError:
            pass
    return model.compute_logits(hidden_states)


def compute_greedy_tokens(
    model,
    hidden_states: torch.Tensor,
    embedding_bias: torch.Tensor | None = None,
    *,
    packed_gather: bool = True,
) -> torch.Tensor:
    """Compute greedy token IDs ([M], int32) with shard-local argmax reduction."""
    if hidden_states.shape[0] == 0:
        return torch.empty((0,), dtype=torch.int32, device=hidden_states.device)

    _, lm_head, logits_processor = _resolve_lm_head_and_processor(model)
    if _supports_shard_local_reduction(lm_head, logits_processor):
        logits, _, tp_size, offset = _compute_shard_logits(
            lm_head, logits_processor, hidden_states, embedding_bias
        )
        max_vals, max_indices = logits.max(dim=-1)
        max_vals = max_vals.to(torch.float32)
        max_indices = max_indices.to(torch.int32) + offset
        if tp_size <= 1:
            return max_indices.reshape(-1)
        if packed_gather:
            local_pair = torch.stack((max_vals, max_indices.to(torch.float32)), dim=-1)
            gathered = _tp_all_gather(local_pair, dim=-1).view(-1, tp_size, 2)
            best_shard = torch.argmax(gathered[:, :, 0], dim=-1, keepdim=True)
            return torch.gather(gathered[:, :, 1], -1, best_shard).squeeze(-1).to(torch.int32)
        gathered_vals = _tp_all_gather(max_vals.unsqueeze(-1), dim=-1)
        gathered_indices = _tp_all_gather(max_indices.unsqueeze(-1), dim=-1)
        best_shard = torch.argmax(gathered_vals, dim=-1, keepdim=True)
        return torch.gather(gathered_indices, -1, best_shard).squeeze(-1).to(torch.int32)

    logits = _compute_full_logits(model, hidden_states, embedding_bias=embedding_bias)
    return logits.argmax(dim=-1).to(torch.int32).reshape(-1)


def compute_topk_tokens(
    model,
    hidden_states: torch.Tensor,
    k: int,
    embedding_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k token IDs ([M, k], int32) and values ([M, k], float32)."""
    k = int(k)
    if k <= 0:
        raise ValueError('topk must be positive')
    if hidden_states.shape[0] == 0:
        return (
            torch.empty((0, k), dtype=torch.int32, device=hidden_states.device),
            torch.empty((0, k), dtype=torch.float32, device=hidden_states.device),
        )

    _, lm_head, logits_processor = _resolve_lm_head_and_processor(model)
    if _supports_shard_local_reduction(lm_head, logits_processor):
        logits, _, tp_size, offset = _compute_shard_logits(
            lm_head, logits_processor, hidden_states, embedding_bias
        )
        local_k = min(k, int(logits.shape[-1]))
        vals, ids = torch.topk(logits, local_k, dim=-1)
        vals = vals.to(torch.float32)
        ids = ids.to(torch.int32) + offset
        if tp_size > 1:
            gathered_vals = _tp_all_gather(vals, dim=-1)
            gathered_ids = _tp_all_gather(ids, dim=-1)
            vals, selected = torch.topk(gathered_vals, k, dim=-1)
            ids = torch.gather(gathered_ids, -1, selected).to(torch.int32)
        return ids, vals

    logits = _compute_full_logits(model, hidden_states, embedding_bias=embedding_bias)
    vals, ids = torch.topk(logits, k, dim=-1)
    return ids.to(torch.int32), vals.to(torch.float32)


def execute_output_spec(
    model,
    hidden_states: torch.Tensor,
    output_spec: OutputSpec,
    embedding_bias: torch.Tensor | None = None,
):
    """Execute OutputSpec over hidden_states without unnecessary logit materialization."""
    if type(output_spec) is not OutputSpec:
        raise ValueError('Invalid OutputSpec')
    if output_spec.mode == 'none':
        return None

    flat_hidden = (
        hidden_states
        if hidden_states.ndim == 2
        else hidden_states.reshape(-1, hidden_states.shape[-1])
    )
    rows = output_spec.rows
    if not rows:
        selected = flat_hidden[:0]
    elif rows == tuple(range(flat_hidden.shape[0])):
        selected = flat_hidden
    else:
        if any(r >= flat_hidden.shape[0] for r in rows):
            raise ValueError('OutputSpec rows exceed hidden_states length')
        row_idx = torch.tensor(rows, dtype=torch.long, device=flat_hidden.device)
        selected = flat_hidden.index_select(0, row_idx)

    if output_spec.mode == 'greedy':
        return compute_greedy_tokens(model, selected, embedding_bias=embedding_bias)
    if output_spec.mode == 'topk':
        return compute_topk_tokens(model, selected, output_spec.topk, embedding_bias=embedding_bias)
    if output_spec.mode == 'logprobs':
        if selected.shape[0] == 0:
            return torch.empty((0, 0), dtype=torch.float32, device=flat_hidden.device)
        logits = _compute_full_logits(model, selected, embedding_bias=embedding_bias)
        return torch.log_softmax(logits.to(torch.float32), dim=-1)
    if output_spec.mode == 'full_logits':
        if selected.shape[0] == 0:
            return torch.empty((0, 0), dtype=flat_hidden.dtype, device=flat_hidden.device)
        return _compute_full_logits(model, selected, embedding_bias=embedding_bias)
    raise ValueError(f'Unsupported output mode: {output_spec.mode}')
