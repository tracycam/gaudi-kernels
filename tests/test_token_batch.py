from itertools import product
from types import SimpleNamespace
import unittest

from gaudi_kernels.engine.token_batch import BatchCapacity, RequestTokens, RequestResult, StepResult, TokenBatch
from gaudi_kernels.serving.executor.scheduled_tokens import from_vllm


class TokenBatchTests(unittest.TestCase):
    def test_mixed_rows_are_compact_and_selected_outputs_keep_query_ownership(self):
        batch = TokenBatch((RequestTokens('decode', (17,), 127, 'decode'),
                            RequestTokens('verify', (17, 17, 17, 17), 126, 'verify'),
                            RequestTokens('prefill', tuple(range(128)), 0, 'prefill')),
                           BatchCapacity(144, 4, 8))
        data = batch.encoded()
        self.assertEqual(batch.num_tokens, 133)
        self.assertEqual(batch.query_start_loc, (0, 1, 5, 133))
        self.assertEqual(data['query_start_loc'], (0, 1, 5, 133, 133))
        self.assertEqual(data['positions'][1:5], (126, 127, 128, 129))
        self.assertEqual(data['row_requests'][5:133], (2,) * 128)
        self.assertEqual(data['row_requests'][133:], (-1,) * 11)
        self.assertEqual(batch.logits_indices, (0, 1, 2, 3, 4))
        self.assertEqual(batch.logits_owners, (('decode', 0), ('verify', 0), ('verify', 1),
                                             ('verify', 2), ('verify', 3)))
        self.assertEqual(data['logits_indices'][5:], (-1,) * 3)

    def test_packing_matches_scalar_query_expansion_under_reorder_and_capacity(self):
        for lengths in product((1, 2, 4, 7), repeat=3):
            requests = tuple(RequestTokens(str(i), tuple(10 + i for _ in range(n)), 125 + i, 'verify')
                             for i, n in enumerate(lengths))
            for ordered in (requests, requests[::-1]):
                rows = sum(lengths)
                batch = TokenBatch(ordered, BatchCapacity(rows + 3, 5, rows + 2))
                expected = [(i, token, request.start_position + j)
                            for i, request in enumerate(ordered) for j, token in enumerate(request.token_ids)]
                data = batch.encoded()
                self.assertEqual(list(zip(data['row_requests'][:rows], data['token_ids'][:rows],
                                          data['positions'][:rows])), expected)
                self.assertEqual(batch.logits_owners, tuple((r.request_id, j) for r in ordered
                                                            for j in range(r.query_length)))

    def test_computed_committed_and_emitted_counts_are_independent(self):
        batch = TokenBatch((RequestTokens('chunk', tuple(range(128)), 0, 'prefill'),
                            RequestTokens('verify', (1, 2, 3, 4), 128, 'verify')))
        result = StepResult((RequestResult('chunk', 128, 128, ()), RequestResult('verify', 4, 2, (2, 9))))
        self.assertIs(batch.validate_result(result), result)
        with self.assertRaisesRegex(ValueError, 'ownership'):
            batch.validate_result(StepResult(result.requests[::-1]))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            batch.validate_result(StepResult((RequestResult('chunk', 1, 1, ()), result.requests[1])))
        with self.assertRaisesRegex(ValueError, 'capacity'):
            batch.validate_result(StepResult((RequestResult('chunk', 128, 128, (5,)), result.requests[1])))

    def test_malformed_and_overflow_work_is_rejected_before_encoding(self):
        with self.assertRaises(ValueError):
            RequestTokens('a', (True,), 0, 'decode')
        with self.assertRaises(ValueError):
            RequestTokens('a', (1, 2), 2**31 - 2, 'verify')
        with self.assertRaises(ValueError):
            RequestTokens('a', (1, 2), 0, 'verify', (1, 0))
        with self.assertRaises(ValueError):
            TokenBatch((RequestTokens('a', (1,), 0, 'decode'),) * 2)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            TokenBatch((RequestTokens('a', (1, 2), 0, 'verify'),), BatchCapacity(1, 1, 2))


class ScheduledTokenTests(unittest.TestCase):
    def fixture(self):
        storage = [[0] * 130, [0] * 132, list(range(130))]
        storage[0][127] = 91
        storage[1][126:130] = [7, 8, 9, 10]
        input_batch = SimpleNamespace(num_reqs=3, req_ids=['d', 'v', 'p'],
            num_computed_tokens_cpu=[127, 126, 0], num_prompt_tokens=[127, 120, 130], token_ids_cpu=storage)
        schedule = SimpleNamespace(num_scheduled_tokens={'d': 1, 'v': 4, 'p': 128},
            total_num_scheduled_tokens=133, scheduled_spec_decode_tokens={'v': [8, 9, 10]})
        requests = {'d': SimpleNamespace(num_tokens=128), 'v': SimpleNamespace(num_tokens=127),
                    'p': SimpleNamespace(num_tokens=130)}
        return input_batch, schedule, requests

    def test_actual_scheduled_storage_packs_known_inputs_and_zero_output_prefill(self):
        inputs, schedule, requests = self.fixture()
        batch = from_vllm(inputs, schedule, requests)
        self.assertEqual(batch.query_lengths, (1, 4, 128))
        self.assertEqual(batch.requests[1].token_ids, (7, 8, 9, 10))
        self.assertEqual(batch.requests[2].output_rows, ())
        schedule.num_scheduled_tokens['p'] = 130
        schedule.total_num_scheduled_tokens = 135
        self.assertEqual(from_vllm(inputs, schedule, requests).requests[2].output_rows, (129,))

    def test_draft_mismatch_and_incomplete_request_coverage_fail(self):
        inputs, schedule, requests = self.fixture()
        inputs.token_ids_cpu[1][128] = 99
        with self.assertRaisesRegex(ValueError, 'draft IDs'):
            from_vllm(inputs, schedule, requests)
        inputs, schedule, requests = self.fixture()
        schedule.num_scheduled_tokens.pop('p')
        with self.assertRaisesRegex(ValueError, 'Scheduled requests'):
            from_vllm(inputs, schedule, requests)
