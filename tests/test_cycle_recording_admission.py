# SPDX-License-Identifier: Apache-2.0
"""Replay must fail before device access when a captured contract is changed."""
from types import SimpleNamespace
import unittest

from gaudi_kernels.engine.token_batch import QueryBatch, RequestQueries
from gaudi_kernels.serving.diagnostics.cycle_recorder import CycleRecorder
from gaudi_kernels.serving.executor.packed_bindings import BoundKVSession, PackedInputBuffers
from gaudi_kernels.serving.executor.packed_pages import PagedKVSession


class CycleRecordingAdmissionTests(unittest.TestCase):
    def test_output_selection_phase_and_storage_changes_fail_before_device_access(self):
        full = QueryBatch((RequestQueries('a', 2, 0, 'verify'),))
        arena = BoundKVSession(PagedKVSession({'layer': (1, 4, 3)}, {'a': 16}, device='cpu'), (2,), 16)
        initial = arena.prepare(full)
        # Do not construct the real HPU recorder. An invalid contract must
        # reject even without an HPU runtime or SDK entry point available.
        recorder = CycleRecorder.__new__(CycleRecorder)
        recorder.inputs = PackedInputBuffers(full, 'cpu')
        recorder.cycle = SimpleNamespace(pending=None, failed=False, target=SimpleNamespace(session=arena))
        recorder.captured = object()
        recorder.signature = recorder._signature(full)
        recorder.addresses = recorder._addresses(initial)
        arena.abort(initial)
        for batch in (QueryBatch((RequestQueries('a', 2, 0, 'verify', (1,)),)),
                      QueryBatch((RequestQueries('a', 2, 0, 'prefill'),))):
            with self.subTest(batch=batch):
                metadata = arena.prepare(batch)
                with self.assertRaises(ValueError):
                    recorder.replay(batch, metadata)
                self.assertIs(arena.pending, metadata)
                arena.abort(metadata)
        metadata = arena.prepare(full)
        recorder.inputs.ids = recorder.inputs.ids.clone()
        with self.assertRaises(ValueError):
            recorder.replay(full, metadata)
        arena.abort(metadata)
        with self.assertRaises(ValueError):
            recorder.replay(full, metadata)
