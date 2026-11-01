import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from gaudi_kernels.engine import ConfigError, EngineConfig
from gaudi_kernels.serving.decode_graph import (
    admitted, plain_greedy, execute, sample_speculative)


class Bias:
    biases = {}


class MinTokens:
    min_toks = {}


def params(**changes):
    return NS(**(dict(temperature=0, repetition_penalty=1, presence_penalty=0,
                     frequency_penalty=0, min_tokens=0, logprobs=None,
                     prompt_logprobs=None) | changes))


class DecodeGraphTest(unittest.TestCase):
    def test_step_signature_excludes_module_kv_and_foreign_state_falls_back(self):
        class Runner:
            def original(self, *args):
                self.seen_caches = args[4]
                return (None, None, None, 'logits')
        runner = Runner()
        runner.kv_caches = [object()]
        runner.input_batch = NS(num_reqs=1, req_ids=['a'])
        runner.batch_changed = False
        runner._prepare_sampling = lambda *args: NS(all_greedy=True, no_penalties=True)
        runner.sampler = lambda **kwargs: NS(logprobs_tensors=None, sampled_token_ids='sample')
        calls = []
        def wrap(fn, **kwargs):
            def program(*args):
                calls.append(args)
                return fn(*args)
            return program
        ht = NS(hpu=NS(wrap_in_hpu_graph_func=wrap))
        modules = {'habana_frameworks': NS(torch=ht), 'habana_frameworks.torch': ht,
                   'vllm.v1.outputs': NS(SamplerOutput=lambda **kw: NS(**kw))}
        tokens, indices, metadata = NS(shape=(1,1)), NS(numel=lambda: 1), object()
        with patch.dict(sys.modules, modules), patch('gaudi_kernels.serving.decode_graph.admitted', return_value=True):
            execute(runner, runner.original, tokens, 'positions', metadata, indices,
                    runner.kv_caches, None, None)
            self.assertEqual(len(calls[0]), 4)
            self.assertIs(runner.seen_caches, runner.kv_caches)
            self.assertEqual(runner._step_graph_sample[1].sampled_token_ids, 'sample')
            foreign = [object()]
            execute(runner, runner.original, tokens, 'positions', metadata, indices,
                    foreign, None, None)
            self.assertEqual(len(calls), 1)
            self.assertIs(runner.seen_caches, foreign)

    def test_sampling_contract_rejects_stateful_or_different_outputs(self):
        self.assertTrue(plain_greedy(params()))
        for change in (dict(temperature=.5), dict(logprobs=0), dict(min_tokens=2),
                       dict(repetition_penalty=1.1), dict(logit_bias={1: 2}),
                       dict(thinking_token_budget=0), dict(bad_words=['word']),
                       dict(allowed_token_ids=[1]), dict(structured_outputs=object())):
            with self.subTest(change=change):
                self.assertFalse(plain_greedy(params(**change)))

    def test_admission_rechecks_processor_state_and_mixed_batches(self):
        runner = NS(speculative_config=None, use_async_scheduling=False,
                    use_structured_output=False, use_aux_hidden_state_outputs=False,
                    _step_decode_count=2,
                    vllm_config=NS(parallel_config=NS(pipeline_parallel_size=1,
                        data_parallel_size=1), lora_config=None, kv_transfer_config=None),
                    input_batch=NS(num_reqs=2, req_ids=['a', 'b'],
                        logitsprocs=NS(non_argmax_invariant=[Bias(), MinTokens()])),
                    requests={key: NS(sampling_params=params()) for key in ('a', 'b')})
        modules = {'vllm.v1.sample.logits_processor.builtin': NS(
            LogitBiasLogitsProcessor=Bias, MinTokensLogitsProcessor=MinTokens)}
        def check():
            return admitted(runner, NS(is_prompt=False), NS(ndim=2, shape=(2, 1)),
                            False, None, None)
        with patch.dict(sys.modules, modules):
            self.assertTrue(check())
            runner.input_batch.logitsprocs.non_argmax_invariant[0].biases = {0: {1: 2}}
            self.assertFalse(check())
            runner.input_batch.logitsprocs.non_argmax_invariant = [object()]
            self.assertFalse(check())
            runner.input_batch.logitsprocs.non_argmax_invariant = []
            runner._step_decode_count = 1
            self.assertFalse(check())
            runner._step_decode_count = 2
            runner.speculative_config = object()
            self.assertFalse(check())

    def test_scope_requires_framework_executor_and_preserves_arithmetic(self):
        base = EngineConfig.from_dict({'runtime': {'executor': 'pytorch', 'runner': 'native'}})
        candidate = EngineConfig.from_dict({'runtime': {'executor': 'pytorch', 'runner': 'native',
                                                        'decode_graph': 'step'}})
        self.assertEqual(base.arithmetic_fingerprint, candidate.arithmetic_fingerprint)
        self.assertEqual(EngineConfig.from_dict({'runtime': {'decode_graph': 'step'}}), candidate)

    def test_speculative_admission_allows_multi_query_and_aux_hidden_states(self):
        runner = NS(speculative_config=NS(num_speculative_tokens=3),
                    use_async_scheduling=False,
                    use_structured_output=False,
                    use_aux_hidden_state_outputs=True,
                    _step_decode_count=2,
                    vllm_config=NS(parallel_config=NS(pipeline_parallel_size=1,
                        data_parallel_size=1), lora_config=None, kv_transfer_config=None),
                    input_batch=NS(num_reqs=2, req_ids=['a', 'b'],
                        logitsprocs=NS(non_argmax_invariant=[])),
                    requests={key: NS(sampling_params=params()) for key in ('a', 'b')})
        modules = {'vllm.v1.sample.logits_processor.builtin': NS(
            LogitBiasLogitsProcessor=Bias, MinTokensLogitsProcessor=MinTokens)}
        def check(rows=8, cols=1):
            return admitted(runner, NS(is_prompt=False), NS(ndim=2, shape=(rows, cols)),
                            False, None, None)
        with patch.dict(sys.modules, modules):
            self.assertTrue(check(rows=2))
            self.assertTrue(check(rows=4))
            self.assertTrue(check(rows=8))
            self.assertTrue(check(rows=16))
            self.assertFalse(check(rows=1))
            self.assertFalse(check(rows=3))
            self.assertFalse(check(rows=8, cols=2))
            runner.speculative_config = NS(num_speculative_tokens=0)
            self.assertFalse(check(rows=2))
            runner.speculative_config = None
            self.assertFalse(check(rows=2))
            runner.use_aux_hidden_state_outputs = False
            self.assertTrue(check(rows=2))
            self.assertFalse(check(rows=8))

    def test_speculative_execute_captures_in_graph_argmax_and_preserves_aux_hidden_states(self):
        import torch
        class Runner:
            def original(self, *args):
                self.seen_caches = args[4]
                logits = torch.tensor([
                    [0.1, 0.9, 0.2],
                    [0.8, 0.1, 0.1],
                    [0.2, 0.3, 0.9],
                    [0.9, 0.4, 0.1],
                ], dtype=torch.float32)
                return ('hidden', ['aux0', 'aux1'], 'sample_hidden', logits)
        runner = Runner()
        runner.kv_caches = [object()]
        runner.speculative_config = NS(num_speculative_tokens=3)
        runner.input_batch = NS(num_reqs=1, req_ids=['req0'])
        runner.batch_changed = True
        runner.sampler = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError('Full sampler must not run inside speculative step graph'))
        calls = []
        def wrap(fn, **kwargs):
            def program(*args):
                calls.append(args)
                return fn(*args)
            return program
        ht = NS(hpu=NS(wrap_in_hpu_graph_func=wrap))
        modules = {'habana_frameworks': NS(torch=ht), 'habana_frameworks.torch': ht,
                   'vllm.v1.outputs': NS(SamplerOutput=lambda **kw: NS(**kw))}
        tokens, indices, metadata = NS(shape=(4, 1)), NS(numel=lambda: 4), object()
        with patch.dict(sys.modules, modules), patch('gaudi_kernels.serving.decode_graph.admitted', return_value=True):
            out = execute(runner, runner.original, tokens, 'positions', metadata, indices,
                          runner.kv_caches, None, None)
            self.assertEqual(len(calls), 1)
            self.assertEqual(out[:3], ('hidden', ['aux0', 'aux1'], 'sample_hidden'))
            self.assertIsNone(runner._step_graph_sample)
            self.assertIsNotNone(runner._step_graph_spec_sample)
            self.assertEqual(runner._step_graph_spec_sample[0], (True, ('req0',)))
            self.assertEqual(runner._step_graph_spec_sample[1].tolist(), [1, 0, 2, 0])

    def test_sample_speculative_uniform_and_variable_draft_verification_and_fallbacks(self):
        import torch
        modules = {'vllm.v1.outputs': NS(SamplerOutput=lambda **kw: NS(**kw))}
        with patch.dict(sys.modules, modules):
            # 1. Uniform batch (batch_size=3, max_spec_len=3):
            # req0: 0 accepted (mismatch at pos 0 -> [g0, -1, -1, -1])
            # req1: 2 accepted (mismatch at pos 2 -> [d0, d1, g2, -1])
            # req2: all 3 accepted (-> [d0, d1, d2, bonus])
            greedy_tokens = torch.tensor([
                99, 11, 12, 13,
                20, 21, 88, 23,
                30, 31, 32, 77,
            ], dtype=torch.int64)
            draft_token_ids = torch.tensor([
                [10], [11], [12],
                [20], [21], [22],
                [30], [31], [32],
            ], dtype=torch.int32)
            spec_meta = NS(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=[3, 3, 3],
                cu_num_draft_tokens=torch.tensor([3, 6, 9], dtype=torch.int32),
                max_spec_len=3,
            )
            sampling_meta = NS(all_greedy=True, no_penalties=True)
            runner = NS(
                _step_graph_spec_sample=((False, ('r0', 'r1', 'r2')), greedy_tokens),
                _prepare_sampling=lambda bc, reqs: sampling_meta,
            )
            fallback_calls = []
            def fallback(*args):
                fallback_calls.append(args)
                return 'fallback_out', 'fallback_meta'

            out, meta = sample_speculative(
                runner, fallback, False, 'logits', ['r0', 'r1', 'r2'], spec_meta)
            self.assertIs(meta, sampling_meta)
            self.assertIsNone(runner._step_graph_spec_sample)
            self.assertEqual(len(fallback_calls), 0)
            self.assertEqual(out.sampled_token_ids.dtype, torch.int32)
            self.assertEqual(out.sampled_token_ids.tolist(), [
                [99, -1, -1, -1],
                [20, 21, 88, -1],
                [30, 31, 32, 77],
            ])

            # 2. Variable draft counts across batch ([3, 1, 0, 2], max_spec_len=3)
            # HPU padded layout: greedy has 4*4=16 entries, draft has 4*3=12 entries
            var_greedy = torch.tensor([
                10, 11, 12, 55,
                20, 66,  0,  0,
                77,  0,  0,  0,
                40, 99, 88,  0,
            ], dtype=torch.int64)
            var_draft = torch.tensor([
                10, 11, 12,
                20,  0,  0,
                 0,  0,  0,
                40, 41,  0,
            ], dtype=torch.int32)
            var_meta = NS(
                draft_token_ids=var_draft,
                num_draft_tokens=[3, 1, 0, 2],
                cu_num_draft_tokens=torch.tensor([3, 4, 4, 6], dtype=torch.int32),
                max_spec_len=3,
            )
            runner._step_graph_spec_sample = ((True, ('a', 'b', 'c', 'd')), var_greedy)
            out_var, _ = sample_speculative(
                runner, fallback, True, 'logits', ['a', 'b', 'c', 'd'], var_meta)
            self.assertEqual(out_var.sampled_token_ids.tolist(), [
                [10, 11, 12, 55],
                [20, 66, -1, -1],
                [77, -1, -1, -1],
                [40, 99, -1, -1],
            ])

            # 3. Packed variable-length layout without num_draft_tokens list
            packed_greedy = torch.tensor([
                10, 11, 12, 55,
                20, 66,
                77,
                40, 99, 88,
            ], dtype=torch.int64)
            packed_draft = torch.tensor([10, 11, 12, 20, 40, 41], dtype=torch.int32)
            packed_meta = NS(
                draft_token_ids=packed_draft,
                cu_num_draft_tokens=torch.tensor([3, 4, 4, 6], dtype=torch.int32),
                max_spec_len=3,
            )
            runner._step_graph_spec_sample = ((False, ('a', 'b', 'c', 'd')), packed_greedy)
            out_packed, _ = sample_speculative(
                runner, fallback, False, 'logits', ['a', 'b', 'c', 'd'], packed_meta)
            self.assertEqual(out_packed.sampled_token_ids.tolist(), [
                [10, 11, 12, 55],
                [20, 66, -1, -1],
                [77, -1, -1, -1],
                [40, 99, -1, -1],
            ])

            # 4. Fallback on cache miss, mismatched request_ids, or non-greedy metadata
            res_miss = sample_speculative(
                runner, fallback, False, 'logits', ['a', 'b', 'c', 'd'], packed_meta)
            self.assertEqual(res_miss, ('fallback_out', 'fallback_meta'))
            self.assertEqual(len(fallback_calls), 1)

            runner._step_graph_spec_sample = ((False, ('a', 'b')), greedy_tokens[:8])
            res_mismatch = sample_speculative(
                runner, fallback, True, 'logits', ['a', 'b'], spec_meta)
            self.assertEqual(res_mismatch, ('fallback_out', 'fallback_meta'))
            self.assertEqual(len(fallback_calls), 2)

            runner._step_graph_spec_sample = ((False, ('r0', 'r1', 'r2')), greedy_tokens)
            runner._prepare_sampling = lambda bc, reqs: NS(all_greedy=False, no_penalties=True)
            res_nongreedy = sample_speculative(
                runner, fallback, False, 'logits', ['r0', 'r1', 'r2'], spec_meta)
            self.assertEqual(res_nongreedy, ('fallback_out', 'fallback_meta'))
            self.assertEqual(len(fallback_calls), 3)

    def test_distributed_greedy_argmax_matches_full_vocab_and_truncates_padding_out_of_place(self):
        import torch
        from gaudi_kernels.engine.token_batch import OutputSpec
        from gaudi_kernels.serving.output_reduction import (
            compute_greedy_tokens,
            execute_output_spec,
        )

        torch.manual_seed(7)
        tp_size = 4
        part_vocab = 16
        padded_vocab = tp_size * part_vocab  # 64
        org_vocab = 60  # Last 4 entries on rank 3 are padding
        hidden_dim = 8
        rows = 5

        hidden = torch.randn(rows, hidden_dim, dtype=torch.float32)
        full_weight = torch.randn(padded_vocab, hidden_dim, dtype=torch.float32)
        full_bias = torch.randn(padded_vocab, dtype=torch.float32)

        # Force the padded entries [60..63] on the last rank to have huge values so
        # in-place/out-of-place padding removal is strictly necessary for correctness.
        with torch.no_grad():
            full_bias[org_vocab:] = 1e6
            # Also create an exact tie between rank 1 (token 20) and rank 2 (token 37) on row 0
            full_weight[0:] = 0.0
            full_bias[:org_vocab] = -10.0
            full_bias[20] = 42.0
            full_bias[37] = 42.0
            full_bias[5] = 19.0
            full_bias[59] = 55.0

        # Compute per-rank shard outputs and verify no shard logit tensor is mutated in-place
        shard_models = []
        observed_shard_logits = {}

        for rank in range(tp_size):
            w_shard = full_weight[rank * part_vocab:(rank + 1) * part_vocab].clone()
            b_shard = full_bias[rank * part_vocab:(rank + 1) * part_vocab].clone()

            def make_apply(r, w):
                def apply(layer, x, bias=None):
                    out = torch.nn.functional.linear(x, w, bias)
                    observed_shard_logits[r] = out.clone()
                    return out
                return apply

            lm_head = NS(
                tp_rank=rank,
                tp_size=tp_size,
                num_embeddings_padded=padded_vocab,
                num_embeddings_per_partition=part_vocab,
                org_vocab_size=org_vocab,
                bias=b_shard,
                quant_method=NS(apply=make_apply(rank, w_shard)),
            )
            logits_processor = NS(
                use_all_gather=True,
                logits_as_input=False,
                org_vocab_size=org_vocab,
                soft_cap=50.0,
                scale=1.5,
                head_dtype=None,
            )
            shard_models.append(
                NS(
                    lm_head=lm_head,
                    logits_processor=logits_processor,
                    compute_logits=lambda *a, **kw: (_ for _ in ()).throw(
                        AssertionError('Full-vocab compute_logits must not be called')
                    ),
                )
            )

        # Test empty batch bypasses lm_head and compute_logits completely
        empty_out = compute_greedy_tokens(shard_models[0], hidden[:0])
        self.assertEqual(empty_out.shape, (0,))
        self.assertEqual(empty_out.dtype, torch.int32)
        self.assertEqual(observed_shard_logits, {})

        # Simulate TP all-gather across all 4 ranks for both packed_gather=True and False
        for packed_mode in (True, False):
            local_inputs = []
            for rank in range(tp_size):
                m = shard_models[rank]
                logits = m.lm_head.quant_method.apply(m.lm_head, hidden, bias=m.lm_head.bias)
                logits = torch.tanh(logits / 50.0) * 50.0 * 1.5
                if rank == tp_size - 1:
                    logits = logits[..., :-(padded_vocab - org_vocab)]
                vals, idx = logits.max(dim=-1)
                vals = vals.to(torch.float32)
                idx = idx.to(torch.int32) + rank * part_vocab
                local_inputs.append((vals, idx))

            call_idx = {'i': 0}

            def fake_all_gather(tensor, dim=-1):
                if packed_mode:
                    pairs = [
                        torch.stack((v, i.to(torch.float32)), dim=-1)
                        for v, i in local_inputs
                    ]
                    return torch.cat(pairs, dim=dim)
                step = call_idx['i']
                call_idx['i'] += 1
                if step % 2 == 0:
                    return torch.cat([v.unsqueeze(-1) for v, _ in local_inputs], dim=dim)
                return torch.cat([i.unsqueeze(-1) for _, i in local_inputs], dim=dim)

            with patch('gaudi_kernels.serving.output_reduction.tensor_model_parallel_all_gather', fake_all_gather):
                # Also test unwrapping through an adapter holder (.model)
                wrapped_model = NS(model=shard_models[3])
                tokens = compute_greedy_tokens(wrapped_model, hidden, packed_gather=packed_mode)
                self.assertEqual(tokens.dtype, torch.int32)
                self.assertEqual(tokens.shape, (rows,))
                # Token 59 (on rank 3, just before the 4 padded entries 60..63) has bias 55.0 > 42.0
                self.assertEqual(tokens.tolist(), [59] * rows)

                # Now test tie-breaking when token 59 is lowered below 42.0:
                # rank 1 (token 20) and rank 2 (token 37) both have 42.0 -> rank 1 (token 20) must win!
                with torch.no_grad():
                    shard_models[3].lm_head.bias[59 - 3 * part_vocab] = -10.0
                local_inputs_tie = []
                for rank in range(tp_size):
                    m = shard_models[rank]
                    logits = m.lm_head.quant_method.apply(m.lm_head, hidden, bias=m.lm_head.bias)
                    logits = torch.tanh(logits / 50.0) * 50.0 * 1.5
                    if rank == tp_size - 1:
                        logits = logits[..., :-(padded_vocab - org_vocab)]
                    vals, idx = logits.max(dim=-1)
                    local_inputs_tie.append((vals.to(torch.float32), idx.to(torch.int32) + rank * part_vocab))
                local_inputs = local_inputs_tie
                call_idx['i'] = 0
                tie_tokens = compute_greedy_tokens(wrapped_model, hidden, packed_gather=packed_mode)
                self.assertEqual(tie_tokens.tolist(), [20] * rows)
                with torch.no_grad():
                    shard_models[3].lm_head.bias[59 - 3 * part_vocab] = 55.0

        # Test execute_output_spec modes ('none', 'greedy', 'topk', 'logprobs', 'full_logits')
        self.assertIsNone(execute_output_spec(shard_models[0], hidden, OutputSpec((), mode='none')))
        single_rank_model = NS(
            compute_logits=lambda h: torch.tensor(
                [[0.1, 0.9, 0.3], [0.8, 0.2, 0.1], [0.2, 0.3, 0.9]], dtype=torch.float32
            )[:h.shape[0]]
        )
        greedy_sel = execute_output_spec(
            single_rank_model, hidden[:3], OutputSpec((0, 2), mode='greedy')
        )
        self.assertEqual(greedy_sel.dtype, torch.int32)
        self.assertEqual(greedy_sel.shape, (2,))
        topk_ids, topk_vals = execute_output_spec(
            single_rank_model, hidden[:2], OutputSpec((0, 1), mode='topk', topk=2)
        )
        self.assertEqual(topk_ids.shape, (2, 2))
        self.assertEqual(topk_vals.shape, (2, 2))
        logprobs = execute_output_spec(
            single_rank_model, hidden[:2], OutputSpec((0, 1), mode='logprobs')
        )
        self.assertEqual(logprobs.shape, (2, 3))

    def test_speculative_step_graph_uses_compute_greedy_tokens_when_full_logits_skipped(self):
        import torch

        class Runner:
            def original(self, *args):
                sample_hidden = torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32
                )
                logits = self.model.compute_logits(sample_hidden)
                return ('hidden', ['aux0'], sample_hidden, logits)

        runner = Runner()
        runner.kv_caches = [object()]
        runner.speculative_config = NS(num_speculative_tokens=3)
        runner.input_batch = NS(num_reqs=1, req_ids=['req0'])
        runner.batch_changed = True
        runner._skip_full_logits_in_spec_graph = True

        weight = torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0], [0.5, 0.5]], dtype=torch.float32
        )
        runner.model = NS(
            lm_head=NS(
                tp_rank=0,
                tp_size=1,
                num_embeddings_padded=4,
                num_embeddings_per_partition=4,
                org_vocab_size=4,
                bias=None,
                quant_method=NS(apply=lambda layer, x, bias=None: torch.nn.functional.linear(x, weight, bias)),
            ),
            logits_processor=NS(
                use_all_gather=True,
                logits_as_input=False,
                org_vocab_size=4,
                soft_cap=None,
                scale=1.0,
                head_dtype=None,
            ),
            compute_logits=lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError('Full compute_logits should be skipped when _skip_full_logits_in_spec_graph=True')
            ),
        )

        ht = NS(hpu=NS(wrap_in_hpu_graph_func=lambda fn, **kw: fn))
        modules = {
            'habana_frameworks': NS(torch=ht),
            'habana_frameworks.torch': ht,
            'vllm.v1.outputs': NS(SamplerOutput=lambda **kw: NS(**kw)),
        }
        tokens, indices, metadata = NS(shape=(4, 1)), NS(numel=lambda: 4), object()
        with patch.dict(sys.modules, modules), patch('gaudi_kernels.serving.decode_graph.admitted', return_value=True):
            out = execute(
                runner, runner.original, tokens, 'positions', metadata, indices,
                runner.kv_caches, None, None
            )
            self.assertIsNone(out[3])
            self.assertIsNotNone(runner._step_graph_spec_sample)
            self.assertEqual(runner._step_graph_spec_sample[1].tolist(), [1, 1, 1, 1])


if __name__ == '__main__':
    unittest.main()

