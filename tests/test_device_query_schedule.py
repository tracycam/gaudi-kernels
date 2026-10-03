# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from gaudi_kernels.engine.token_batch import BatchCapacity, QueryBatch, RequestQueries, RequestTokens, TokenBatch
from gaudi_kernels.serving.executor.packed_attention import PackedKVSession
from gaudi_kernels.serving.executor.packed_bindings import PackedInputBuffers


class DeviceQueryTests(unittest.TestCase):
    def test_same_schedule_without_fabricated_host_token_ids(self):
        cpu = TokenBatch((RequestTokens('a', (10,), 0, 'decode'),
                          RequestTokens('b', (20, 21, 22, 23), 0, 'verify')), BatchCapacity(8, 3, 6))
        schedule = cpu.query_schedule()
        self.assertEqual(schedule.query_start_loc, cpu.query_start_loc)
        self.assertEqual(schedule.logits_owners, cpu.logits_owners)
        self.assertEqual(schedule.capacity_key, cpu.capacity_key)
        self.assertFalse(hasattr(schedule.requests[0], 'token_ids'))
        with self.assertRaises(ValueError):
            schedule.encoded()
        inputs = PackedInputBuffers(schedule, 'cpu')
        with self.assertRaises(ValueError):
            inputs.validate(schedule)
        arena = PackedKVSession({'layer': (1, 4, 3)}, 16, device='cpu')
        metadata = arena.prepare(schedule)
        ids = torch.tensor([10, 20, 21, 22, 23], dtype=torch.int32)
        positions = torch.cat(metadata.query_positions)
        inputs.update_device(schedule, ids, positions)
        inputs.validate(schedule)
        self.assertTrue(torch.equal(inputs.ids, ids))
        self.assertTrue(torch.equal(inputs.positions, positions))
        arena.commit(metadata, (1, 2))
        self.assertEqual(tuple(len(arena.committed[r]) for r in schedule.request_ids), (1, 2))

    def test_invalid_schedule_extent_and_binding_fail_closed(self):
        for args in (('a', 0, 0, 'verify'), ('a', 2, 0, 'decode'), ('a', 2, -1, 'verify')):
            with self.assertRaises(ValueError):
                RequestQueries(*args)
        batch = QueryBatch((RequestQueries('a', 2, 0, 'verify'),))
        inputs = PackedInputBuffers(batch, 'cpu')
        with self.assertRaises(ValueError):
            inputs.update_device(batch, torch.ones(3, dtype=torch.int32), torch.arange(2))
        with self.assertRaises(ValueError):
            inputs.validate(batch)
