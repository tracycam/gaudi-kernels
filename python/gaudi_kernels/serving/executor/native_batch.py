"""Named-input native B>1/T1 execution through the owned vLLM runner.

This reuses the qualified operator implementations, not an expanded-weight
MoE replacement. Prefill and speculative KV transactions remain separate gates.
"""
import ctypes
import json
import math
import time

from gaudi_kernels.engine.context import context
from .named_inputs import NamedInputs, InputBinding


def install(module, cls, *, bridge_sample, state, reset, vote, api, holder, roles, root):
    import torch
    import torch.distributed as dist
    import habana_frameworks.torch.core as hc
    api.native_multi_prepare.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    api.native_multi_input_bytes.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
    api.native_multi_compare_input.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64]
    api.native_multi_read_output.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    api.native_multi_step.argtypes = [ctypes.POINTER(InputBinding), ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    b1_sample, b1_execute = cls.sample_tokens, cls.execute_model

    def eligible(self, grammar):
        b, sched = self.input_batch, self.scheduler_output
        if not context().native.active or self.warmup_mode or sched is None:
            return False
        if not 1 < b.num_reqs <= 8 or b.num_reqs not in context().selection.engine.runtime.buckets.batch:
            return False
        if self.speculative_config or self.use_structured_output or grammar is not None:
            return False
        if (self.use_async_scheduling or self.vllm_config.max_concurrent_batches != 1 or
                self.vllm_config.lora_config or module.has_kv_transfer_group() or
                self.parallel_config.data_parallel_size != 1 or self.parallel_config.pipeline_parallel_size != 1):
            return False
        if (self.uses_mrope or self.num_mamba_like_layers or self.model_has_chunked_attention or
                not self.interleaved_sliding_window or self.use_contiguous_pa or
                getattr(getattr(self, 'defragmenter', None), 'enabled', False)):
            return False
        for i, rid in enumerate(b.req_ids[:b.num_reqs]):
            req = self.requests[rid]
            p = req.sampling_params
            if (sched.num_scheduled_tokens.get(rid) != 1 or self._is_prompt(i, sched) or req.mm_features or
                    p is None or p.temperature != 0 or p.repetition_penalty != 1 or
                    p.presence_penalty != 0 or p.frequency_penalty != 0 or getattr(p, 'min_tokens', 0)):
                return False
            if any(getattr(p, name, None) for name in ('logprobs', 'prompt_logprobs', 'allowed_token_ids',
                    'bad_words_token_ids', 'logits_processors', 'logit_bias', 'structured_outputs')):
                return False
            pos = int(b.num_computed_tokens_cpu[i])
            if int(b.num_tokens_no_spec[i]) != pos + 1:
                raise RuntimeError('Native batch input-token ownership mismatch')
        return True

    def collect(self):
        roles.collect = {}
        try:
            self._prepare_inputs(self.scheduler_output, 0, self.input_batch.num_reqs)
            return roles.collect
        finally:
            roles.collect = None

    def read_output(plan):
        storage = (ctypes.c_ubyte * plan['output_bytes'])()
        rc = api.native_multi_read_output(storage, len(storage))
        if rc:
            raise RuntimeError('NATIVE_BATCH_READ_FAILED ' + str(rc))
        values = torch.frombuffer(bytearray(storage), dtype=plan['output_dtype']).tolist()
        return values

    def capture(self):
        s = state(self)
        hc.mark_step(); torch.hpu.synchronize()
        roles.reset(root)
        if api.e1_begin(torch.hpu.current_device()):
            raise RuntimeError('NATIVE_BATCH_CAPTURE_BEGIN_FAILED')
        s['capturing'] = True
        try:
            result = bridge_sample(self, None)
            hc.mark_step(); api.e1_pause(); torch.hpu.synchronize()
        finally:
            api.e1_pause(); s['capturing'] = False
        tag = f"batch-phase{s['phase']}-capture{len(s['captures'])}-rank{dist.get_rank()}"
        rec = {'kind': 'batch', 'batch': self.input_batch.num_reqs, 'status': 'REJECTED'}
        rc = api.e1_end(torch.hpu.current_device(), str(root / (tag+'-commands.txt')).encode())
        held = (ctypes.c_uint64 * 3)()
        holder.e1_tensor_hold_stats(held, str(root / (tag+'-storage.txt')).encode())
        rec.update(record_code=rc, storage_lease_counts=list(held))
        if not vote(rc == 0 and held[0] > 0 and held[2] > 0):
            reset(self)
            raise RuntimeError('NATIVE_BATCH_CAPTURE_FAILED ' + json.dumps(rec))
        output_bytes = ctypes.c_uint64()
        rc = api.native_multi_prepare(ctypes.byref(output_bytes))
        rec['prepare_code'] = rc
        plan, error = None, None
        try:
            if rc:
                raise ValueError('SDK binding rejected: ' + str(rc))
            outputs = [r for r in roles.records if r.get('direction') == 'output']
            if len(outputs) != 1 or outputs[0]['role'] != 'sampled_tokens':
                raise ValueError('One explicit sampled-token output required')
            out = outputs[0]
            numel = math.prod(out['shape'])
            dtype = getattr(torch, out['dtype'].removeprefix('torch.'))
            if dtype not in (torch.int32, torch.int64):
                raise ValueError('Sampled tokens must be integer IDs')
            if output_bytes.value == numel * 4:
                dtype = torch.int32
            elif output_bytes.value != numel * 8 or dtype != torch.int64:
                raise ValueError('Unexpected sampled-output encoding')
            plan = {'kind': 'batch', 'key': tuple(self.input_batch.req_ids[:self.input_batch.num_reqs]),
                'inputs': NamedInputs(api, roles.records), 'output_bytes': output_bytes.value,
                'output_dtype': dtype, 'capture_index':len(s['captures'])}
        except ValueError as exc:
            error = str(exc)
        rec['binding_error'] = error
        if not vote(plan is not None):
            reset(self)
            (root / (tag+'.json')).write_text(json.dumps(rec, indent=2)+'\n')
            raise RuntimeError('NATIVE_BATCH_BIND_FAILED ' + json.dumps(rec))
        tensors = {name: s['owners'][i] for name, i in (('hidden',0), ('selected',2), ('logits',3))}
        before = {k: v.detach().cpu().clone() for k, v in tensors.items()}
        times = [ctypes.c_uint64() for _ in range(3)]
        rc = api.e1_replay(1, *[ctypes.byref(t) for t in times])
        checks = {k: torch.equal(before[k].contiguous().view(torch.uint8),
            v.detach().cpu().contiguous().view(torch.uint8)) for k,v in tensors.items()}
        checks['samples'] = [[v] for v in read_output(plan)[:self.input_batch.num_reqs]] == result.sampled_token_ids
        rec.update(replay_code=rc, checks=checks, submission_count=api.e1_submission_count())
        (root / (tag+'-semantics.json')).write_text(json.dumps(roles.records, indent=2)+'\n')
        if not vote(rc == 0 and all(checks.values())):
            reset(self)
            (root / (tag+'.json')).write_text(json.dumps(rec, indent=2)+'\n')
            raise RuntimeError('NATIVE_BATCH_GATE_FAILED ' + json.dumps(rec))
        rec['status'] = 'NUMERICAL_REPLAY_PASS'
        s['captures'].append(rec); s['plan'] = plan
        (root / (tag+'.json')).write_text(json.dumps(rec, indent=2)+'\n')
        return result

    @torch.inference_mode()
    def sample(self, grammar):
        s = state(self)
        plan = s['plan']
        if not eligible(self, grammar):
            if plan is not None and plan.get('kind') == 'batch':
                reset(self)
            return b1_sample(self, grammar)
        key = tuple(self.input_batch.req_ids[:self.input_batch.num_reqs])
        if plan is not None and (plan.get('kind') != 'batch' or plan['key'] != key):
            reset(self); plan = None
        if plan is not None:
            begin = time.perf_counter_ns()
            try:
                bindings, owners = plan['inputs'].bind(collect(self))
            except ValueError:
                reset(self); plan = None
        if plan is None:
            return capture(self)
        metadata_end = time.perf_counter_ns()
        positions = [int(self.input_batch.num_computed_tokens_cpu[i]) for i in range(len(key))]
        output = (ctypes.c_ubyte * plan['output_bytes'])()
        times = (ctypes.c_uint64 * 4)()
        rc = api.native_multi_step(bindings, len(bindings), output, len(output), times)
        if rc:
            raise RuntimeError('NATIVE_BATCH_STEP_FAILED ' + str(rc))
        replay_end = time.perf_counter_ns()
        values = torch.frombuffer(bytearray(output), dtype=plan['output_dtype']).tolist()[:len(key)]
        b = self.input_batch
        for i, (rid, value) in enumerate(zip(key, values)):
            index = int(b.num_tokens_no_spec[i])
            if index + 1 > self.max_model_len + 1:
                raise RuntimeError('Native batch exceeds model length')
            b.token_ids_cpu[i, index] = value
            b.num_tokens_no_spec[i] = b.num_tokens[i] = index + 1
            self.requests[rid].output_token_ids.append(value)
        self.scheduler_output = None; self.warmup_mode = False; self.invalid_req_indices = []
        end = time.perf_counter_ns()
        s['native_steps'].append(dict(kind='batch', batch=len(key), tokens=values,
            positions=positions, capture_index=plan['capture_index'],
            native_wall_ns=times[0], enqueue_ns=times[1], synlaunch_ns=times[2], hccl_ns=times[3],
            cpu_metadata_ns=metadata_end-begin, cpu_commit_ns=end-replay_end,
            api_detail_valid=context().native.replay_mode!='compact',
            caller_begin_ns=begin, caller_end_ns=end, caller_ns=end-begin))
        return module.ModelRunnerOutput(req_ids=list(key), req_id_to_index=dict(b.req_id_to_index),
            sampled_token_ids=[[v] for v in values], logprobs=None, prompt_logprobs_dict={}, pooler_output=[])

    def execute(self, *args, **kwargs):
        begin = time.perf_counter_ns()
        count = len(state(self)['native_steps'])
        result = b1_execute(self, *args, **kwargs)
        if result is None and context().native.single_rpc and eligible(self, None):
            state(self)['single_rpc_steps'] += 1
            result = self.sample_tokens(None)
        if len(state(self)['native_steps']) > count:
            end = time.perf_counter_ns()
            state(self)['native_steps'][-1].update(worker_begin_ns=begin, worker_end_ns=end, worker_wall_ns=end-begin)
        return result

    cls.sample_tokens, cls.execute_model = sample, execute
