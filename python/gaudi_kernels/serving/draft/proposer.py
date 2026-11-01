# SPDX-License-Identifier: Apache-2.0
"""HPU DFlash proposer over vLLM's target sampling and request lifecycle.

The existing checkpoint reference is deliberately reused. This first serving
backend establishes request/KV semantics; it does not claim a fused cycle or
the performance of the old fixed-cohort diagnostic executor.
"""
import ctypes
from pathlib import Path

import torch
from torch import nn

from gaudi_kernels.serving.draft.dflash import load_checkpoint
from gaudi_kernels.serving.draft.requests import ContextPool, prompt_slices
from gaudi_kernels.serving.output_reduction import compute_greedy_tokens


class ProposalProgram(nn.Module):
    """One tensor program; request ownership and target KV stay outside it."""
    _accepts_raw_decode_inputs = True

    def __init__(self, draft, context, target, proposals):
        super().__init__()
        self.draft, self.context, self.proposals = draft, context, proposals
        # The shared target is already owned/moved by vLLM. Do not register the
        # entire target as another drafter submodule or duplicate its loading.
        object.__setattr__(self, 'target', target)
        dev = getattr(context, 'device', 'cpu')
        self.register_buffer(
            '_block_arange',
            torch.arange(draft.spec.block, dtype=torch.int32, device=dev)[None],
            persistent=False,
        )

    @property
    def target_model(self):
        return self.target

    def forward(self, features, query_positions, counts, slots, anchors, next_positions, proposal_count=None):
        b = anchors.shape[0]
        if query_positions.ndim != 2 or query_positions.shape[0] != b:
            query_positions = query_positions.reshape(b, -1)
        if isinstance(features, (tuple, list)):
            features = features[0] if len(features) == 1 else torch.cat(features, -1)
        if features.ndim != 3 or features.shape[0] != b:
            features = features.reshape(b, query_positions.shape[1], -1)
        projected = self.draft.project_context(features, query_positions)
        self.context.commit_slots(projected, query_positions, counts, slots)
        chosen = self.proposals if proposal_count is None else proposal_count
        if chosen == 0:
            return torch.full((b, self.proposals), -1, dtype=torch.int32, device=anchors.device)
        context, context_positions, valid = self.context.read(slots)
        block_arange = (self._block_arange
                        if self._block_arange.device == next_positions.device
                        else torch.arange(self.draft.spec.block, dtype=torch.int32, device=next_positions.device)[None])
        positions = next_positions[:, None] + block_arange
        noise = self.draft.noise(self.target.embed_input_ids(anchors))
        hidden = self.draft(noise, positions, context, context_positions, valid)
        masked = hidden[:, 1:1 + chosen].reshape(-1, self.draft.spec.hidden)
        if hasattr(self.draft, 'hidden_projection') and callable(self.draft.hidden_projection):
            masked = self.draft.hidden_projection(masked)
        output = compute_greedy_tokens(self.target, masked).reshape(b, chosen)
        if chosen == self.proposals:
            return output
        return torch.cat((output, torch.full((b, self.proposals - chosen), -1,
                                             dtype=torch.int32, device=anchors.device)), 1)


class HpuDFlashProposer:
    def __init__(self, vllm_config, device, model_runner, *, row_budget=0):
        config = vllm_config.speculative_config
        self.directory = Path(config.draft_model_config.model)
        self.num_speculative_tokens = config.num_speculative_tokens
        if not 1 <= self.num_speculative_tokens <= 7:
            raise ValueError('Checkpoint DFlash supports 1..7 parallel proposals')
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError('HPU DFlash auxiliary features currently require PP=1')
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError('HPU DFlash needs prompt features; prefix-cache feature recovery is not implemented')
        self.device = device
        self.runner = model_runner
        self.max_cached_graphs = model_runner.graph_options().max_cached_graphs
        self.capacity = vllm_config.scheduler_config.max_num_seqs
        self.row_budget = row_budget
        self.proposal_lengths = {}
        self.calls = 0
        self.proposed_tokens = 0
        self.committed_decode_queries = 0
        self._meta_buffers = {}
        self._rows_buffers = {}
        self._prefill_meta = None

    def load_model(self, target_model):
        from vllm.distributed import (get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
                                      tensor_model_parallel_all_reduce, tensor_model_parallel_all_gather)
        self.target = target_model
        self.draft = load_checkpoint(
            self.directory, self.device, tp_rank=get_tensor_model_parallel_rank(),
            tp_size=get_tensor_model_parallel_world_size(), reduce_sum=tensor_model_parallel_all_reduce,
            gather_output=tensor_model_parallel_all_gather)
        self.draft.fold_gqa = True
        self.context = ContextPool(self.draft, self.capacity, self.device)
        self.model = ProposalProgram(self.draft, self.context, self.target, self.num_speculative_tokens)

    def configure_target(self, target_model):
        from vllm.model_executor.models.interfaces import SupportsEagle3
        # Public mixin unwraps multimodal language-model holders correctly.
        # Its indices name layer boundaries; checkpoint IDs name decoder layers.
        SupportsEagle3.set_aux_hidden_state_layers(
            target_model, tuple(layer + 1 for layer in self.draft.spec.target_layers))

    def wrap_model(self):
        import habana_frameworks.torch as ht
        if ht.utils.internal.is_lazy():
            self.model = ht.hpu.wrap_in_hpu_graph(self.model, disable_tensor_cache=False, asynchronous=False,
                                                max_graphs=self.max_cached_graphs)
            self.model._accepts_raw_decode_inputs = True

    def release_requests(self, request_ids):
        if hasattr(self, 'context'):
            self.context.release(request_ids)

    def _get_meta_buffers(self, batch_size):
        buffers = self.__dict__.get('_meta_buffers')
        if buffers is None:
            buffers = self._meta_buffers = {}
        key = (batch_size, str(self.device))
        entry = buffers.get(key)
        if entry is None:
            cpu_buf = torch.empty((4, batch_size), dtype=torch.int32, device='cpu')
            c_arr = (ctypes.c_int32 * (4 * batch_size)).from_address(cpu_buf.data_ptr())
            if torch.device(self.device).type == 'cpu':
                dev_buf = cpu_buf
            else:
                dev_buf = torch.empty((4, batch_size), dtype=torch.int32, device=self.device)
            entry = (cpu_buf, c_arr, dev_buf, dev_buf[0], dev_buf[1], dev_buf[2], dev_buf[3])
            buffers[key] = entry
        return entry

    def _get_prefill_meta_buffers(self):
        entry = self.__dict__.get('_prefill_meta')
        dev_key = str(self.device)
        if entry is None or entry[0] != dev_key:
            cpu_buf = torch.empty((2, 1), dtype=torch.int32, device='cpu')
            c_arr = (ctypes.c_int32 * 2).from_address(cpu_buf.data_ptr())
            if torch.device(self.device).type == 'cpu':
                dev_buf = cpu_buf
            else:
                dev_buf = torch.empty((2, 1), dtype=torch.int32, device=self.device)
            entry = (dev_key, cpu_buf, c_arr, dev_buf, dev_buf[0], dev_buf[1])
            self._prefill_meta = entry
        return entry

    def _get_rows_buffer(self, rows):
        n = len(rows)
        buffers = self.__dict__.get('_rows_buffers')
        if buffers is None:
            buffers = self._rows_buffers = {}
        key = (n, str(self.device))
        entry = buffers.get(key)
        if entry is None:
            cpu_buf = torch.empty(n, dtype=torch.int64, device='cpu')
            c_arr = (ctypes.c_int64 * n).from_address(cpu_buf.data_ptr())
            if torch.device(self.device).type == 'cpu':
                dev_buf = cpu_buf
            else:
                dev_buf = torch.empty(n, dtype=torch.int64, device=self.device)
            entry = (cpu_buf, c_arr, dev_buf)
            buffers[key] = entry
        cpu_buf, c_arr, dev_buf = entry
        if n == 1:
            c_arr[0] = rows[0]
        else:
            c_arr[:] = rows
        if dev_buf is not cpu_buf:
            dev_buf.copy_(cpu_buf, non_blocking=True)
        return dev_buf

    def observe_prefill(self, request_ids, positions, aux_hidden_states, query_counts, *, merged):
        if not any(self.request_limit(request_id) for request_id in request_ids):
            return
        if aux_hidden_states is None or len(aux_hidden_states) != len(self.draft.spec.target_layers):
            raise RuntimeError('DFlash requires all checkpoint target feature layers')
        if isinstance(query_counts, torch.Tensor) and query_counts.device.type != 'cpu':
            sched = getattr(getattr(self.runner, 'scheduler_output', None), 'num_scheduled_tokens', None)
            if isinstance(sched, dict) and all(rid in sched for rid in request_ids):
                query_counts = [int(sched[rid]) for rid in request_ids]
            else:
                query_counts = [int(x) for x in query_counts.detach().cpu().reshape(-1).tolist()]
        flat_aux = tuple(x.reshape(-1, self.draft.spec.hidden) for x in aux_hidden_states)
        flat_positions = positions.reshape(-1)
        window = self.draft.spec.window
        is_hpu = torch.device(self.device).type != 'cpu'
        batch = getattr(self.runner, 'input_batch', None)
        req_id_to_index = getattr(batch, 'req_id_to_index', None)
        computed_cpu = getattr(batch, 'num_computed_tokens_cpu', None)
        prompt_tokens = getattr(batch, 'num_prompt_tokens', None)
        _, cpu_buf, c_arr, dev_buf, count_t, slot_t = self._get_prefill_meta_buffers()
        for request_id, rows in zip(request_ids, prompt_slices(positions, query_counts, merged=merged)):
            if not self.request_limit(request_id):
                continue
            q_len = int(rows.stop - rows.start)
            if q_len <= 0:
                continue
            slot_id = self.context.slot(request_id)
            if (isinstance(req_id_to_index, dict) and computed_cpu is not None and
                    prompt_tokens is not None and request_id in req_id_to_index):
                b_idx = req_id_to_index[request_id]
                if int(prompt_tokens[b_idx]) - (int(computed_cpu[b_idx]) + q_len) >= window:
                    continue
            if is_hpu and not merged and q_len < window and positions.shape[1] >= window and (rows.start + window) <= flat_positions.numel():
                win_rows = slice(rows.start, rows.start + window)
                pos = flat_positions[win_rows][None]
                valid_count = q_len
            else:
                win_rows = rows
                pos = flat_positions[rows][-window:][None]
                valid_count = int(pos.shape[1])
            if not valid_count:
                continue
            if len(flat_aux) == 1:
                source = flat_aux[0][win_rows][-window:][None]
            else:
                source = torch.cat(tuple(x[win_rows][-window:] for x in flat_aux), -1)[None]
            projected = self.draft.project_context(source, pos)
            c_arr[0] = valid_count
            c_arr[1] = slot_id
            if dev_buf is not cpu_buf:
                dev_buf.copy_(cpu_buf, non_blocking=True)
            self.context.commit_slots(projected, pos, count_t, slot_t)

    def _propose(self, request_ids, anchors, next_positions, features, query_positions, counts):
        batch_size = len(request_ids)
        if not batch_size:
            return torch.empty((0, self.num_speculative_tokens), dtype=torch.int32, device='cpu')
        slots = [self.context.slot(request_id) for request_id in request_ids]
        chosen = self.proposal_count(len(self.runner.input_batch.req_ids), request_ids)
        cpu_buf, c_arr, dev_buf, counts_t, slots_t, ids_t, next_pos_t = self._get_meta_buffers(batch_size)
        if batch_size == 1:
            c_arr[0] = counts[0]
            c_arr[1] = slots[0]
            c_arr[2] = anchors[0]
            c_arr[3] = next_positions[0]
        else:
            c_arr[:] = (*counts, *slots, *anchors, *next_positions)
        if dev_buf is not cpu_buf:
            dev_buf.copy_(cpu_buf, non_blocking=True)
        output = self.model(features, query_positions, counts_t, slots_t, ids_t, next_pos_t, chosen)
        self.calls += 1
        self.proposal_lengths = {request_id: min(chosen, self.request_limit(request_id)) for request_id in request_ids}
        self.proposed_tokens += sum(self.proposal_lengths.values())
        return output

    def request_limit(self, request_id):
        params = self.runner.requests[request_id].sampling_params
        return (params.extra_args or {}).get('draft_token_limit', self.num_speculative_tokens)

    def proposal_count(self, requests, request_ids=None):
        maximum = (max(self.request_limit(request_id) for request_id in request_ids)
                   if request_ids else self.num_speculative_tokens)
        return min(maximum, max(0, self.row_budget // requests - 1)) if self.row_budget else maximum

    def format_proposals(self, request_ids, rows):
        # The scheduler must receive actual lengths; padding -1 is a tensor
        # transport detail, not a token that may enter target embeddings.
        result = [row[:self.proposal_lengths.get(request_id, 0)]
                  for request_id, row in zip(request_ids, rows)]
        self.proposal_lengths = {}
        return result

    def propose_from_target(self, *, sampled_token_ids, aux_hidden_states, num_decodes,
                            prefill_data, decode_data, scheduler_output, **unused):
        """Consume vLLM's accepted outputs, never draft-only target features."""
        request_ids = self.runner.input_batch.req_ids
        self.proposal_lengths = {}
        total_requests = len(request_ids)
        limits = [self.request_limit(request_id) for request_id in request_ids]
        if not any(limits):
            return torch.full((total_requests, self.num_speculative_tokens), -1,
                              dtype=torch.int32, device='cpu')
        active_ids, anchors, positions, output_rows, counts = [], [], [], [], []
        decode_active_indices = []
        stride = 1
        computed_cpu = self.runner.input_batch.num_computed_tokens_cpu
        num_sched = scheduler_output.num_scheduled_tokens
        if decode_data is not None:
            if aux_hidden_states is None or len(aux_hidden_states) != len(self.draft.spec.target_layers):
                raise RuntimeError('Missing target decode auxiliary features')
            # HPU target queries are flattened to [padded_B*T, 1]. Their
            # last tensor dimension is NOT the per-request query stride.
            stride = max(num_sched[request_ids[i]] for i in range(num_decodes))
            total_decode_rows = num_decodes * stride
            if decode_data.position_ids.numel() < total_decode_rows:
                raise RuntimeError('Target decode rows do not match scheduled query layout')
            for i in range(num_decodes):
                if not limits[i]:
                    continue
                emitted = sampled_token_ids[i]
                if not emitted:
                    continue
                # One committed query for each emitted token (accepted drafts
                # plus the target's bonus/correction); rejected suffix ignored.
                count = len(emitted)
                request_id = request_ids[i]
                scheduled = num_sched[request_id]
                if count > scheduled or count > stride:
                    raise RuntimeError('Accepted target prefix exceeds scheduled queries')
                decode_active_indices.append(i)
                counts.append(count)
                self.committed_decode_queries += count
                start = int(computed_cpu[i])
                active_ids.append(request_id)
                anchors.append(emitted[-1])
                positions.append(start + count)
                output_rows.append(i)

        prefill_active_count = 0
        # Every prompt chunk was observed at its producer, even a chunk without
        # logits. Only completed prompts have an anchor and need proposals now.
        for i in range(num_decodes, total_requests):
            if not limits[i]:
                continue
            emitted = sampled_token_ids[i]
            if not emitted:
                continue
            request_id = request_ids[i]
            prefill_active_count += 1
            active_ids.append(request_id)
            anchors.append(emitted[-1])
            positions.append(int(computed_cpu[i]) + num_sched[request_id])
            output_rows.append(i)
            counts.append(0)

        if not active_ids:
            return torch.full((total_requests, self.num_speculative_tokens), -1,
                              dtype=torch.int32, device='cpu')

        total_hidden = len(self.draft.spec.target_layers) * self.draft.spec.hidden
        # Fast path: only decode requests are active and they form a contiguous prefix 0..k-1
        if prefill_active_count == 0 and len(decode_active_indices) == (decode_active_indices[-1] + 1):
            k = len(decode_active_indices)
            active_rows = k * stride
            pos_ids = decode_data.position_ids
            if getattr(self.model, '_accepts_raw_decode_inputs', False):
                expected_numel = active_rows * self.draft.spec.hidden
                if len(aux_hidden_states) == 1:
                    x0 = aux_hidden_states[0]
                    features_in = x0 if x0.numel() == expected_numel else x0.reshape(-1, self.draft.spec.hidden)[:active_rows]
                else:
                    features_in = tuple(
                        x if x.numel() == expected_numel else x.reshape(-1, self.draft.spec.hidden)[:active_rows]
                        for x in aux_hidden_states
                    )
                pos_in = pos_ids if pos_ids.numel() == active_rows else pos_ids.reshape(-1)[:active_rows]
            else:
                features_in = torch.cat(aux_hidden_states, -1).reshape(-1, total_hidden)[:active_rows].reshape(k, stride, total_hidden)
                pos_in = pos_ids.reshape(-1)[:active_rows].reshape(k, stride)
        else:
            feature_rows, position_rows = [], []
            if decode_active_indices:
                features = torch.cat(aux_hidden_states, -1).reshape(-1, total_hidden)
                target_positions = decode_data.position_ids.reshape(-1)
                for i in decode_active_indices:
                    rows = slice(i * stride, (i + 1) * stride)
                    feature_rows.append(features[rows])
                    position_rows.append(target_positions[rows])
            if prefill_active_count:
                zero_feat = torch.zeros(stride, total_hidden, dtype=torch.bfloat16, device=self.device)
                neg_pos = torch.full((stride,), -1, dtype=torch.int32, device=self.device)
                for _ in range(prefill_active_count):
                    feature_rows.append(zero_feat)
                    position_rows.append(neg_pos)
            features_in = torch.stack(feature_rows)
            pos_in = torch.stack(position_rows)

        proposed = self._propose(active_ids, anchors, positions, features_in, pos_in, counts)
        if len(active_ids) == total_requests:
            return proposed
        output = torch.full((total_requests, self.num_speculative_tokens), -1,
                            dtype=proposed.dtype, device=proposed.device)
        output.index_copy_(0, self._get_rows_buffer(output_rows), proposed)
        return output
