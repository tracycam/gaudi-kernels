# SPDX-License-Identifier: Apache-2.0
"""Bounded full-cycle command recording, for exclusive synchronous diagnostics.

This is not scheduler admission or an explicit Synapse graph builder. The
recorder retains all captured temporary storage. Only fixed request/query
extents and stable mutable inputs may be replayed; output delivery is fenced.
"""
import ctypes
import time
from dataclasses import replace


class CycleRecorder:
    def __init__(self, coordinator, inputs, metadata, root, rank, vote):
        from gaudi_kernels.engine.context import context
        self.cycle, self.inputs, self.vote = coordinator, inputs, vote
        self.root, self.rank = root, rank
        self.signature = self._signature(metadata.batch)
        self.addresses = self._addresses(metadata)
        self.api = ctypes.CDLL(None)
        self.api.e1_begin.argtypes = [ctypes.c_int]
        self.api.e1_end.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.api.e1_replay.argtypes = [ctypes.c_uint, *[ctypes.POINTER(ctypes.c_uint64)]*3]
        self.api.e1_submission_count.restype = ctypes.c_uint32
        self.api.e1_set_semantic_role.argtypes = [ctypes.c_char_p]
        self.holder = ctypes.CDLL(context().path('tensor_hold', 'host'))
        self.holder.e1_tensor_hold_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p]
        self.captured = None

    @staticmethod
    def _signature(schedule):
        return (schedule.request_ids, schedule.query_lengths, schedule.logits_indices,
                tuple(request.kind for request in schedule.requests))

    def _addresses(self, metadata):
        return tuple(t.data_ptr() for t in (self.inputs.ids, self.inputs.positions,
            metadata.slot_mapping, *metadata.visible_slots, *metadata.query_positions,
            *metadata.visible_lengths, metadata.window_page_ids,
            metadata.window_page_groups, metadata.flat_query_positions))

    def capture(self, schedule, metadata, anchors, remaining):
        import torch
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
        torch.hpu.synchronize()
        self.api.e1_set_semantic_role(None)
        if not self.vote(self.api.e1_begin(torch.hpu.current_device()) == 0):
            raise RuntimeError('Full-cycle recording begin failed')
        try:
            result = self.cycle.run_prepared(schedule, metadata, anchors, remaining, inputs=self.inputs)
            # The target diagnostic drains before verification/context append.
            # Flush those trailing commands before closing the recording too.
            htcore.mark_step()
            torch.hpu.synchronize()
        finally:
            self.api.e1_pause()
        code = self.api.e1_end(torch.hpu.current_device(), str(self.root/f'rank{self.rank}-commands.txt').encode())
        held = (ctypes.c_uint64*3)()
        self.holder.e1_tensor_hold_stats(held, str(self.root/f'rank{self.rank}-storage.txt').encode())
        self.record = dict(record_code=code, sdk_submission_count=self.api.e1_submission_count(),
                           storage_lease_counts=list(held), intermediate_python_on_replay=False,
                           explicit_graph_producer=False)
        if not self.vote(code == 0 and held[0] > 0 and held[2] > 0):
            raise RuntimeError('Full-cycle recording did not retain all-rank storage')
        self.captured = result
        return result

    def replay(self, schedule, metadata, *, profile=False):
        if (self.captured is None or self.cycle.pending is not None or self.cycle.failed or
                self.cycle.target.session.pending is not metadata or metadata.batch is not schedule or
                self._signature(schedule) != self.signature or
                self._addresses(metadata) != self.addresses):
            raise ValueError('Recorded cycle binding/ownership changed')
        import torch
        import habana_frameworks.torch.core as htcore
        drain_begin = time.perf_counter_ns()
        htcore.mark_step()
        torch.hpu.synchronize()
        replay_begin = time.perf_counter_ns()
        profiling = None
        if profile:
            if self.rank == 0:
                profiling = self._profile_start()
            # The optional profiler must be ready before any TP peer queues
            # the observed cycle. This diagnostic barrier is not serving work.
            self.vote(True)
            replay_begin = time.perf_counter_ns()
        times = [ctypes.c_uint64() for _ in range(3)]
        code = self.api.e1_replay(1, *[ctypes.byref(t) for t in times])
        replay_end = time.perf_counter_ns()
        if not self.vote(code == 0):
            self.cycle.failed = True
            raise RuntimeError('Full-cycle recorded replay failed')
        vote_end = time.perf_counter_ns()
        if profile:
            if self.rank == 0 and profiling.get('start_code') == 0:
                counts = (ctypes.c_uint64*2)()
                profiling['stop_code'] = self.api.e1_profile_stop(
                    str(self.root/f'rank{self.rank}-device-trace.jsonl').encode(), counts)
                profiling.update(trace_bytes=counts[0], trace_events=counts[1])
            self.vote(True)
        # Device output storage is retained by the recorder. Only the Python
        # transaction descriptor changes; no model/drafter/verification code
        # runs here and no token value crosses the host before delivery.
        target = replace(self.captured.target, transaction=metadata, output_owners=schedule.logits_owners)
        result = replace(self.captured, target=target)
        self.cycle.pending = result
        return result, dict(replay_code=code, enqueue_host_ns=times[0].value,
                            launch_host_ns=times[1].value, hccl_host_ns=times[2].value,
                            binding_drain_wall_ns=replay_begin-drain_begin,
                            sdk_replay_with_completion_wall_ns=replay_end-replay_begin,
                            diagnostic_rank_vote_wall_ns=vote_end-replay_end,
                            profiled=profile, profile=profiling)

    def _profile_start(self):
        import torch
        result = {}
        try:
            self.api.e1_profile_required.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
            self.api.e1_profile_start.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
            self.api.e1_profile_stop.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
            required = ctypes.c_uint32()
            result['query_code'] = self.api.e1_profile_required(ctypes.byref(required))
            result['required_bytes'] = required.value
            if result['query_code']:
                return result
            # Keep the buffer alive until Stop/GetTrace finishes. Profiling is
            # optional and errors are evidence, never a device-reset trigger.
            self.profile_storage = torch.empty(max(required.value, 512), dtype=torch.uint8,
                                               device=self.cycle.target.runner.device)
            result['start_code'] = self.api.e1_profile_start(self.profile_storage.data_ptr(),
                                                            self.profile_storage.numel())
        except Exception as error:
            result['error'] = repr(error)
        return result

    def release(self):
        self.api.e1_pause()
        return self.holder.native_tensor_hold_release()
