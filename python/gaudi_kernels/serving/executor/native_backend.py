"""Opt-in committing native backend for synchronous B1 greedy vLLM decode.

The first eligible step records and gates the actual bridge recipe sequence.
Later eligible steps return native-generated tokens and commit vLLM state.
No comparison rollout or restore runs on the fast path.
"""
from gaudi_kernels.engine.context import context as execution_context
import ctypes, json, os, time
from pathlib import Path

def install(module, *, runner_cls=None):
    if execution_context().selection.engine.runtime.executor == 'pytorch':
        return
    import torch
    import torch.distributed as dist
    import habana_frameworks.torch.core as hc
    cls = runner_cls or module.HPUModelRunner
    api = ctypes.CDLL(None)
    api.e1_set_replay_mode.argtypes = [ctypes.c_int]
    api.e1_last_device_span.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    api.e1_submission_count.restype = ctypes.c_uint32
    api.e1_begin.argtypes = [ctypes.c_int]
    api.e1_end.argtypes = [ctypes.c_int, ctypes.c_char_p]
    api.e1_set_semantic_role.argtypes = [ctypes.c_char_p]
    api.e1_replay.argtypes = [ctypes.c_uint, *[ctypes.POINTER(ctypes.c_uint64)] * 3]
    api.native_plan_prepare.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    api.native_step.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64)]
    api.native_plan_capacities.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    if execution_context().native.pad_window:
        api.native_plan_enable_padded_window.argtypes = []
    api.native_step_bound.argtypes = [ctypes.c_int32] * 3 + [ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32] * 2 + [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64)]
    api.e1_sampled_id.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    api.e1_profile_required.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    api.e1_profile_start.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    api.e1_profile_stop.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
    holder = ctypes.CDLL(execution_context().path('tensor_hold', 'host'))
    assert holder.e1_tensor_hold_install() == 0
    holder.e1_tensor_hold_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p]
    root = Path(str(execution_context().startup.run_dir.parent)) / execution_context().startup.run_dir.name
    from gaudi_kernels.serving.executor.executor_inputs import install as install_roles
    if runner_cls is None:
        roles = install_roles(module, api, root)
    else:
        from gaudi_kernels.serving.executor.executor_inputs import SemanticTransfers
        roles = SemanticTransfers(api, root)
        cls._native_transfers = roles
    original_sample = cls.sample_tokens
    original_model = cls._execute_model_generic
    original_execute = cls.execute_model

    def state(self):
        if not hasattr(self, '_native_backend'):
            self._native_backend = {'plan': None, 'owners': None, 'capturing': False, 'captures': [], 'native_steps': [], 'fallbacks': {}, 'phase': 0, 'single_rpc_steps': 0}
        return self._native_backend

    def audit_boundary(self, pos, outputs, mode):
        requested = ','.join((str(v) for v in execution_context().native.boundary_positions))
        if not requested or pos not in {int(v) for v in requested.split(',')}:
            return
        import hashlib
        s = state(self)
        destination = root / 'boundary-audit'
        destination.mkdir(exist_ok=True)
        path = destination / f"phase{s['phase']}-rank{dist.get_rank()}-position{pos}-{mode}.pt"
        if path.exists():
            raise RuntimeError('Duplicate boundary audit output')
        tensors = {k: outputs[i].detach().cpu().contiguous() for (k, i) in [('hidden', 0), ('selected', 2), ('logits', 3)]}
        torch.save(tensors, path)
        record = dict(position=pos, mode=mode, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()) for (k, v) in tensors.items()})
        s.setdefault('boundary_audit', []).append(record)

    def vote(ok):
        from vllm.distributed import get_tp_group
        v = torch.tensor([int(ok)], dtype=torch.int32, device='cpu')
        dist.all_reduce(v, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
        return bool(v.item())

    def reset(self):
        s = state(self)
        assert holder.native_tensor_hold_release() == 0, 'Native plan drain/release failed'
        s['plan'] = None
        s['owners'] = None

    def supported(self, grammar):
        sched = self.scheduler_output
        b = self.input_batch
        if not execution_context().native.active:
            return (None, 'disabled')
        if sched is None or self.warmup_mode:
            return (None, 'no_live_work')
        if b.num_reqs != 1:
            return (None, 'batch_not_one')
        if self.use_async_scheduling:
            return (None, 'async_scheduling')
        if self.vllm_config.max_concurrent_batches != 1:
            return (None, 'batch_queue')
        if self.speculative_config or self.use_structured_output or grammar is not None:
            return (None, 'sampling_mode')
        if self.vllm_config.lora_config or module.has_kv_transfer_group():
            return (None, 'lora_or_kv_transfer')
        rid = b.req_ids[0]
        req = self.requests[rid]
        p = req.sampling_params
        if p is None or p.temperature != 0 or p.repetition_penalty != 1 or (p.presence_penalty != 0) or (p.frequency_penalty != 0):
            return (None, 'sampling_parameters')
        if p.logprobs is not None or p.prompt_logprobs is not None:
            return (None, 'logprobs')
        for field in ('logprobs', 'prompt_logprobs', 'allowed_token_ids', 'bad_words_token_ids', 'logits_processors', 'logit_bias', 'structured_outputs'):
            if getattr(p, field, None):
                return (None, 'sampling_' + field)
        if getattr(p, 'min_tokens', 0) != 0 or req.mm_features:
            return (None, 'minimum_tokens_or_multimodal')
        if sched.num_scheduled_tokens.get(rid) != 1 or self._is_prompt(0, sched):
            return (None, 'prefill_or_catchup')
        pos = int(b.num_computed_tokens_cpu[0])
        reuse = execution_context().native.reuse_pages
        if reuse and self.use_contiguous_pa:
            return (None, 'contiguous_pa')
        if reuse and self.sliding_window != 128:
            return (None, 'unsupported_window')
        if getattr(getattr(self, 'defragmenter', None), 'enabled', False):
            return (None, 'defragmenter')
        if self.parallel_config.data_parallel_size != 1 or self.parallel_config.pipeline_parallel_size != 1:
            return (None, 'dp_or_pp')
        ready = state(self)['plan'] is not None
        if pos % 128 == 127 and (not (ready and state(self)['plan'].get('padded_window', False))):
            return (None, 'window_one_page')
        if pos % 128 < 2 and (not reuse or not ready):
            return (None, 'page_transition')
        if int(b.num_tokens_no_spec[0]) != pos + 1:
            raise RuntimeError('Native input-token ownership mismatch')
        token = int(b.token_ids_cpu[0, pos])
        group = self._get_attention_group_id_for_hybrid()
        block = int(b.block_table[group].get_cpu_tensor()[0, pos // 128])
        slot = int(self._resolve_block(block)) * 128 + pos % 128
        key = (rid, tuple((tuple(x) for x in req.block_ids))) if reuse else (rid, pos // 128, tuple((tuple(x) for x in req.block_ids)))
        return ((key, token, pos, slot), None)

    def execute(self, *args, **kwargs):
        out = original_model(self, *args, **kwargs)
        s = state(self)
        if s['capturing']:
            s['owners'] = out
        if ','.join((str(v) for v in execution_context().native.boundary_positions)):
            metadata = args[2] if len(args) > 2 else kwargs['attn_metadata']
            if self.input_batch.num_reqs == 1 and (not metadata.is_prompt):
                audit_boundary(self, int(self.input_batch.num_computed_tokens_cpu[0]), out, 'bridge')
        return out
    cls._execute_model_generic = execute

    @torch.inference_mode()
    def sample(self, grammar_output):
        s = state(self)
        (context, reason) = supported(self, grammar_output)
        if context is None:
            keep = reason == 'window_one_page' and execution_context().native.reuse_pages and (s['plan'] is not None) and (s['plan']['key'][0] == self.input_batch.req_ids[0])
            position = int(self.input_batch.num_computed_tokens_cpu[0]) if keep else None
            if s['plan'] is not None and (not keep):
                reset(self)
            s['fallbacks'][reason] = s['fallbacks'].get(reason, 0) + 1
            result = original_sample(self, grammar_output)
            if keep:
                s['plan']['bridge_completed_position'] = position
            return result
        (key, token, pos, slot) = context
        reuse = execution_context().native.reuse_pages
        if s['plan'] is not None and s['plan']['key'] != key:
            if not reuse or s['plan']['key'][0] != key[0]:
                reset(self)
        pages = window = None
        if s['plan'] is not None and reuse and (s['plan']['page'] != pos // 128 or s['plan']['key'] != key or (s['plan'].get('padded_window', False) and pos % 128 == 127)):
            group = self._get_attention_group_id_for_hybrid()
            n = pos // 128 + 1
            ids = self.input_batch.block_table[group].get_cpu_tensor()[0, :n].tolist()
            win = module._window_block_tables([ids], [pos], self.sliding_window, 128)[0]
            sizes = (self.bucketing_manager.find_decode_bucket(1, len(ids))[2], s['plan']['capacities'][1] if s['plan'].get('padded_window', False) else self.bucketing_manager.find_decode_bucket(1, len(win))[2])
            if tuple(s['plan']['capacities']) != sizes:
                reset(self)
            else:
                pages = (ctypes.c_int32 * len(ids))(*ids)
                window = (ctypes.c_int32 * len(win))(*win)
        if s['plan'] is not None:
            output = ctypes.c_int32()
            timings = (ctypes.c_uint64 * 4)()
            profile = s.get('device_profile')
            if profile and profile['remaining'] and (not profile.get('attempted')):
                profile['attempted'] = True
                from habana_frameworks.torch.utils.experimental import _data_ptr
                storage = profile['storage']
                rc = api.e1_profile_start(int(_data_ptr(storage)), storage.numel())
                profile['start_code'] = rc
                profile['started'] = rc == 0
                if rc:
                    profile['remaining'] = 0
                from vllm.distributed import get_tp_group
                dist.barrier(group=get_tp_group().cpu_group)
            start = time.perf_counter_ns()
            code = api.native_step_bound(token, pos, slot, pages, len(pages) if pages is not None else 0, window, len(window) if window is not None else 0, s['plan'].get('bridge_completed_position', -1), ctypes.byref(output), timings) if reuse else api.native_step(token, pos, slot, ctypes.byref(output), timings)
            if code:
                raise RuntimeError(f'NATIVE_STEP_FAILED code={code} position={pos}')
            audit_boundary(self, pos, s['owners'], 'native')
            if profile and profile.get('started'):
                profile['positions'].append(pos)
                profile['remaining'] -= 1
                if profile['remaining'] == 0:
                    counts = (ctypes.c_uint64 * 2)()
                    profile['stop_code'] = api.e1_profile_stop(str(profile['path']).encode(), counts)
                    (profile['trace_bytes'], profile['entries']) = list(counts)
                    profile['started'] = False
            value = output.value
            b = self.input_batch
            rid = b.req_ids[0]
            self.scheduler_output = None
            self.warmup_mode = False
            index = int(b.num_tokens_no_spec[0])
            assert index == pos + 1
            b.token_ids_cpu[0, index] = value
            b.num_tokens_no_spec[0] = index + 1
            b.num_tokens[0] = index + 1
            self.requests[rid].output_token_ids.append(value)
            s['plan'].update(key=key, page=pos // 128, bridge_completed_position=-1)
            self.invalid_req_indices = []
            result = module.ModelRunnerOutput(req_ids=[rid], req_id_to_index=dict(b.req_id_to_index), sampled_token_ids=[[value]], logprobs=None, prompt_logprobs_dict={}, pooler_output=[])
            end = time.perf_counter_ns()
            record = dict(position=pos, token=value, native_wall_ns=timings[0], enqueue_ns=timings[1], synlaunch_ns=timings[2], hccl_ns=timings[3], caller_ns=end - start, caller_begin_ns=start, caller_end_ns=end, api_detail_valid=execution_context().native.replay_mode!='compact')
            if execution_context().native.replay_mode == 'device':
                span = ctypes.c_uint64()
                if api.e1_last_device_span(ctypes.byref(span)) != 0:
                    raise RuntimeError('Missing diagnostic device event span')
                record['device_stream_span_ns'] = span.value
            s['native_steps'].append(record)
            return result
        if pos % 128 < 2 or pos % 128 == 126:
            return original_sample(self, grammar_output)
        hc.mark_step()
        torch.hpu.synchronize()
        roles.reset(root)
        assert api.e1_begin(torch.hpu.current_device()) == 0
        s['capturing'] = True
        try:
            result = original_sample(self, grammar_output)
            hc.mark_step()
            api.e1_pause()
            torch.hpu.synchronize()
        finally:
            api.e1_pause()
            s['capturing'] = False
        rank = dist.get_rank()
        number = len(s['captures'])
        tag = f"native-phase{s['phase']}-capture{number}-rank{rank}"
        code = api.e1_end(torch.hpu.current_device(), str(root / (tag + '-commands.txt')).encode())
        rec = {'position': pos, 'record_code': code, 'status': 'REJECTED', 'scope': 'identical-recipe executor replay; not generation acceptance'}
        held = (ctypes.c_uint64 * 3)()
        holder.e1_tensor_hold_stats(held, str(root / (tag + '-storage.txt')).encode())
        rec['storage_lease_counts'] = list(held)
        if not vote(code == 0 and held[0] > 0 and (held[2] > 0)):
            s['captures'].append(rec)
            reset(self)
            (root / (tag + '.json')).write_text(json.dumps(rec, indent=2) + '\n')
            raise RuntimeError('NATIVE_CAPTURE_FAILED ' + json.dumps(rec))
        owners = s['owners']
        tensors = {k: owners[i] for (k, i) in [('hidden', 0), ('selected', 2), ('logits', 3)]}
        before = {k: v.detach().cpu().clone() for (k, v) in tensors.items()}
        times = [ctypes.c_uint64() for _ in range(3)]
        rc = api.e1_replay(1, *[ctypes.byref(t) for t in times])
        sampled = ctypes.c_int32()
        sc = api.e1_sampled_id(ctypes.byref(sampled))
        checks = {k: torch.equal(before[k].contiguous().view(torch.uint8), v.detach().cpu().contiguous().view(torch.uint8)) for (k, v) in tensors.items()}
        checks['sample'] = sc == 0 and sampled.value == result.sampled_token_ids[0][0]
        rec.update(replay_code=rc, checks=checks)
        if not vote(rc == 0 and all(checks.values())):
            s['captures'].append(rec)
            (root / (tag + '.json')).write_text(json.dumps(rec, indent=2) + '\n')
            reset(self)
            raise RuntimeError('NATIVE_GATE_FAILED ' + json.dumps(rec))
        info = (ctypes.c_int64 * 5)()
        code = api.native_plan_prepare(info)
        rec.update(prepare_code=code, plan_info=list(info))
        if not vote(code == 0):
            s['captures'].append(rec)
            (root / (tag + '.json')).write_text(json.dumps(rec, indent=2) + '\n')
            reset(self)
            raise RuntimeError('NATIVE_BIND_FAILED ' + json.dumps(rec))
        assert info[0] == pos and info[1] == slot
        capacities = (ctypes.c_int64 * 2)()
        assert api.native_plan_capacities(capacities) == 0
        rec['capacities'] = list(capacities)
        padded = execution_context().native.pad_window and reuse and (capacities[1] >= 2) and (getattr(self, '_production_config').decode.attention.swa != 'vendor')
        if padded and api.native_plan_enable_padded_window() != 0:
            raise RuntimeError('NATIVE_PADDED_WINDOW_ENABLE_FAILED')
        rec['padded_window'] = padded
        rec['submission_count'] = api.e1_submission_count()
        rec['status'] = 'NUMERICAL_REPLAY_PASS'
        s['captures'].append(rec)
        s['plan'] = {'key': key, 'page': pos // 128, 'capacities': list(capacities), 'bridge_completed_position': -1, 'padded_window': padded}
        (root / (tag + '.json')).write_text(json.dumps(rec, indent=2) + '\n')
        return result
    cls.sample_tokens = sample

    def execute_model(self, *args, **kwargs):
        begin = time.perf_counter_ns()
        count = len(state(self)['native_steps'])
        result = original_execute(self, *args, **kwargs)
        if result is None and execution_context().native.single_rpc:
            (context, _) = supported(self, None)
            if context is not None:
                state(self)['single_rpc_steps'] += 1
                result = self.sample_tokens(None)
        if len(state(self)['native_steps']) > count:
            end = time.perf_counter_ns()
            state(self)['native_steps'][-1].update(worker_begin_ns=begin, worker_end_ns=end, worker_wall_ns=end-begin)
        return result
    cls.execute_model = execute_model
    cls._native_reset = reset
    if runner_cls is not None and execution_context().selection.engine.runtime.batch_replay:
        from .native_batch import install as install_batch
        install_batch(module, cls, bridge_sample=original_sample, state=state, reset=reset,
                      vote=vote, api=api, holder=holder, roles=roles, root=root)
    print('NATIVE_BACKEND_INSTALLED synchronous_b1_greedy=1', flush=True)

def configure(worker, active, single_rpc=False, reuse_pages=False, replay_mode='compact'):
    if execution_context().selection.engine.runtime.executor == 'pytorch':
        raise ValueError('External native replay is absent in PyTorch execution')
    modes = {'compact': 0, 'legacy': 1, 'detail': 2, 'device': 3}
    if replay_mode not in modes:
        raise ValueError('Unknown replay mode')
    import torch
    import habana_frameworks.torch.core as hc
    hc.mark_step()
    torch.hpu.synchronize()
    runner = worker.model_runner
    if hasattr(runner, '_native_reset'):
        runner._native_reset()
    api = ctypes.CDLL(None)
    api.e1_set_replay_mode.argtypes = [ctypes.c_int]
    if api.e1_set_replay_mode(modes[replay_mode]) != 0:
        raise RuntimeError('Replay mode change requires an empty drained plan')
    s = getattr(runner, '_native_backend', None)
    if s is not None:
        profile = s.get('device_profile')
        if profile and profile.get('started'):
            api = ctypes.CDLL(None)
            counts = (ctypes.c_uint64 * 2)()
            api.e1_profile_stop.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
            profile['stop_code'] = api.e1_profile_stop(str(profile['path']).encode(), counts)
            (profile['trace_bytes'], profile['entries']) = list(counts)
            profile['started'] = False
        s.pop('device_profile', None)
        s['phase'] += 1
        s['captures'] = []
        s['native_steps'] = []
        s['fallbacks'] = {}
        s['single_rpc_steps'] = 0
        s['boundary_audit'] = []
    execution_context().native.active = active
    execution_context().native.single_rpc = single_rpc
    execution_context().native.reuse_pages = reuse_pages
    execution_context().native.replay_mode = replay_mode
    held = (ctypes.c_uint64 * 3)()
    holder = ctypes.CDLL(execution_context().path('tensor_hold', 'host'))
    holder.e1_tensor_hold_stats(held, None)
    assert list(held) == [0, 0, 0], 'Storage leases remain after drain/reset'
    return {'active': active, 'pid': os.getpid(), 'released_lease_counts': list(held)}

def summary(worker):
    s = getattr(worker.model_runner, '_native_backend', {})
    profile = s.get('device_profile')
    from gaudi_kernels.serving.diagnostics.host_clock import clock_domain
    return {'pid': os.getpid(), 'rank': worker.rank, 'clock_domain': clock_domain(), 'phase': s.get('phase'), 'captures': s.get('captures', []), 'native_steps': s.get('native_steps', []), 'fallbacks': s.get('fallbacks', {}), 'single_rpc_steps': s.get('single_rpc_steps', 0), 'boundary_audit': s.get('boundary_audit', []), 'device_profile': None if profile is None else {k: str(v) if isinstance(v, Path) else v for (k, v) in profile.items() if k != 'storage'}}

def configure_profile(worker, count=3):
    """Arm a separate native diagnostic run; no profile is included in TPS."""
    import torch
    import habana_frameworks.torch.core as hc
    if count not in (1, 2, 3) or not execution_context().native.active:
        raise ValueError('profile requires native replay and 1..3 separate diagnostic steps')
    api = ctypes.CDLL(None)
    api.e1_profile_required.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    api.e1_profile_start.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    api.e1_profile_stop.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
    required = ctypes.c_uint32()
    code = api.e1_profile_required(ctypes.byref(required))
    if code:
        return {'rank': worker.rank, 'query_code': code, 'armed': False}
    if required.value > 512 * 1024 * 1024:
        raise ValueError('profiler buffer exceeds bounded 512MiB budget')
    storage = torch.empty(max(required.value, 4096), dtype=torch.uint8, device='hpu')
    hc.mark_step()
    torch.hpu.synchronize()
    s = worker.model_runner._native_backend
    if s.get('device_profile'):
        raise RuntimeError('a profile is already armed')
    path = Path(str(execution_context().startup.run_dir.parent)) / execution_context().startup.run_dir.name / f'native-device-profile-rank{worker.rank}.jsonl'
    if path.exists():
        raise FileExistsError(path)
    s['device_profile'] = {'storage': storage, 'path': path, 'remaining': count, 'positions': [], 'required_bytes': required.value, 'scope': 'separate profiled diagnostic; excluded from TPS'}
    return {'rank': worker.rank, 'query_code': code, 'armed': True, 'required_bytes': required.value}
