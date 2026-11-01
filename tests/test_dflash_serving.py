# SPDX-License-Identifier: Apache-2.0
"""Serving ownership, rejection and proposal alignment without a model load."""
from types import SimpleNamespace
import unittest

import torch

from gaudi_kernels.serving.draft.proposer import HpuDFlashProposer, ProposalProgram
from gaudi_kernels.serving.draft.requests import ContextPool, prompt_slices


class FakeDraft:
    spec = SimpleNamespace(layers=1, window=4, kv_heads=1, dim=2, hidden=2, target_layers=(0,), block=4)

    def __init__(self):
        self.projected = []

    def project_context(self, features, positions):
        self.projected.append(positions.clone())
        values = features.unsqueeze(2)
        return ((values, values.clone()),)

    def noise(self, anchors):
        return torch.cat((anchors[:, None], torch.zeros(anchors.shape[0], 3, 2)), 1)

    def __call__(self, noise, positions, *context):
        return positions[:, :, None].expand(-1, -1, 2).to(torch.bfloat16)


class DFlashServingTest(unittest.TestCase):
    def proposer(self, ids=('a', 'b')):
        draft = FakeDraft()
        p = HpuDFlashProposer.__new__(HpuDFlashProposer)
        p.device = 'cpu'
        p.draft = p.model = draft
        p.num_speculative_tokens = 2
        p.row_budget = 0
        p.proposal_lengths = {}
        p.calls = p.proposed_tokens = p.committed_decode_queries = 0
        p.context = ContextPool(draft, 3, 'cpu')
        p.runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=list(ids), num_computed_tokens_cpu=[4, 14]))
        p.runner.requests = {request_id: SimpleNamespace(sampling_params=SimpleNamespace(extra_args=None))
                             for request_id in ids}
        p.target = SimpleNamespace(embed_input_ids=lambda ids: ids[:, None].expand(-1, 2).to(torch.bfloat16),
                                   compute_logits=lambda hidden: -(torch.arange(32)[None] - hidden[:, :1].float()).abs())
        p.model = ProposalProgram(draft, p.context, p.target, p.num_speculative_tokens)
        return p

    def state(self, p, names):
        return p.context.read(torch.tensor([p.context.slot(name) for name in names], dtype=torch.int32))

    def commit(self, p, name, positions):
        pos = torch.tensor(positions, dtype=torch.int32)[-p.draft.spec.window:][None]
        projected = p.draft.project_context(pos[:, :, None].expand(-1, -1, 2).to(torch.bfloat16), pos)
        p.context.commit_slots(projected, pos, torch.tensor([pos.shape[1]], dtype=torch.int32),
                               torch.tensor([p.context.slot(name)], dtype=torch.int32))

    def test_reorder_release_and_same_id_new_lifetime(self):
        p = self.proposer()
        self.commit(p, 'a', [0, 1])
        self.commit(p, 'b', [10, 11])
        _, pos, valid = self.state(p, ['b', 'a'])
        self.assertEqual(pos[0][valid[0]].tolist(), [10, 11])
        self.assertEqual(pos[1][valid[1]].tolist(), [0, 1])
        p.release_requests(['a'])
        _, pos, valid = self.state(p, ['a'])
        self.assertFalse(valid.any())
        self.assertIn('b', p.context.owners)

    def test_rejected_suffix_never_overwrites_live_ring(self):
        p = self.proposer(('a',))
        self.commit(p, 'a', [0, 1, 2, 3])
        decode = SimpleNamespace(position_ids=torch.arange(4, 8)[None])
        features = torch.arange(4, 8)[:, None].expand(-1, 2).to(torch.bfloat16)
        p.propose_from_target(sampled_token_ids=[[20, 21]], aux_hidden_states=[features], num_decodes=1,
                              prefill_data=None, decode_data=decode,
                              scheduler_output=SimpleNamespace(num_scheduled_tokens={'a': 4}))
        pairs, pos, valid = self.state(p, ['a'])
        self.assertEqual(sorted(pos[valid].tolist()), [2, 3, 4, 5])
        self.assertEqual(sorted(pairs[0][0][0, :, 0, 0].float().tolist()), [2, 3, 4, 5])
        self.assertEqual(p.committed_decode_queries, 2)

    def test_ragged_acceptance_no_cross_request_features(self):
        p = self.proposer()
        decode = SimpleNamespace(position_ids=torch.tensor([[4, 5, 6, 7], [14, 15, 16, 17]]))
        features = decode.position_ids.reshape(-1, 1).expand(-1, 2).to(torch.bfloat16)
        output = p.propose_from_target(sampled_token_ids=[[20], [30, 31, 32]], aux_hidden_states=[features],
            num_decodes=2, prefill_data=None, decode_data=decode,
            scheduler_output=SimpleNamespace(num_scheduled_tokens={'a': 4, 'b': 4}))
        # Proposals begin one position AFTER each known anchor, row zero excluded.
        self.assertEqual(output.tolist(), [[6, 7], [18, 19]])
        self.assertEqual([x.tolist() for x in p.draft.projected], [[[4, 5, 6, 7], [14, 15, 16, 17]]])

    def test_flattened_hpu_verify_layout_uses_scheduled_stride(self):
        p = self.proposer()
        positions = torch.tensor([4, 5, 6, 7, 14, 15, 16, 17])[:, None]
        output = p.propose_from_target(sampled_token_ids=[[20, 21], [30]],
            aux_hidden_states=[positions.expand(-1, 2).to(torch.bfloat16)], num_decodes=2,
            prefill_data=None, decode_data=SimpleNamespace(position_ids=positions),
            scheduler_output=SimpleNamespace(num_scheduled_tokens={'a': 4, 'b': 1}))
        self.assertEqual(output.tolist(), [[7, 8], [16, 17]])
        self.assertEqual([x.tolist() for x in p.draft.projected], [[[4, 5, 6, 7], [14, 15, 16, 17]]])

    def test_intermediate_prompt_chunks_and_trailing_window(self):
        p = self.proposer()
        for positions in (torch.tensor([[0, 1, 2, 999]]), torch.tensor([[3, 4, 5, 999]])):
            features = positions.reshape(-1, 1).expand(-1, 2).to(torch.bfloat16)
            p.observe_prefill(['a'], positions, [features], [3], merged=False)
        _, pos, valid = self.state(p, ['a'])
        self.assertEqual(sorted(pos[valid].tolist()), [2, 3, 4, 5])
        self.commit(p, 'b', list(range(10, 20)))
        self.assertEqual(p.draft.projected[-1].tolist(), [[16, 17, 18, 19]])

    def test_merged_prompt_does_not_admit_padding(self):
        p = self.proposer()
        pos = torch.tensor([[0, 1, 10, 11, 12, 999]])
        features = pos.reshape(-1, 1).expand(-1, 2).to(torch.bfloat16)
        p.observe_prefill(['a', 'b'], pos, [features], [2, 3], merged=True)
        self.assertEqual([x.tolist() for x in p.draft.projected], [[[0, 1]], [[10, 11, 12]]])

    def test_finished_or_incomplete_request_has_no_proposal(self):
        p = self.proposer()
        output = p.propose_from_target(sampled_token_ids=[[], [7]], aux_hidden_states=None,
            num_decodes=0, prefill_data=None, decode_data=None,
            scheduler_output=SimpleNamespace(num_scheduled_tokens={'a': 3, 'b': 2}))
        self.assertEqual(output.tolist(), [[-1, -1], [17, 18]])
        self.assertEqual(p.calls, 1)

    def test_reject_bad_layout_and_accepted_prefix(self):
        with self.assertRaises(ValueError):
            prompt_slices(torch.zeros(1, 4), [3, 3], merged=True)
        p = self.proposer(('a',))
        with self.assertRaises(RuntimeError):
            p.propose_from_target(sampled_token_ids=[[1, 2, 3]], aux_hidden_states=[torch.zeros(2, 2)],
                num_decodes=1, prefill_data=None, decode_data=SimpleNamespace(position_ids=torch.zeros(1, 2)),
                scheduler_output=SimpleNamespace(num_scheduled_tokens={'a': 2}))

    def test_adaptive_budget_returns_real_lengths_to_scheduler(self):
        p = self.proposer()
        p.num_speculative_tokens = 7
        p.row_budget = 8
        self.assertEqual([p.proposal_count(b) for b in (1, 2, 3, 4, 8)], [7, 3, 1, 1, 0])
        p.proposal_lengths = {'a': 3, 'b': 1}
        self.assertEqual(p.format_proposals(['b', 'a', 'finished'], [[5, -1], [7, 8, 9, -1], [-1]]),
                         [[5], [7, 8, 9], []])
        self.assertEqual(p.proposal_lengths, {})

    def test_zero_drafts_still_commits_context_without_calling_draft(self):
        p = self.proposer()
        pos = torch.tensor([[4, 5]])
        slot = torch.tensor([p.context.slot('a')], dtype=torch.int32)
        output = p.model(pos[:, :, None].expand(-1, -1, 2).to(torch.bfloat16), pos,
                         torch.tensor([1]), slot, torch.tensor([20]), torch.tensor([5]), 0)
        self.assertEqual(output.tolist(), [[-1, -1]])
        _, stored, valid = self.state(p, ['a'])
        self.assertEqual(stored[valid].tolist(), [4])

    def test_request_limit_reaches_actual_scheduler_lengths(self):
        p = self.proposer()
        p.runner.requests['a'].sampling_params.extra_args = {'draft_token_limit': 1}
        positions = torch.tensor([4, 5, 14, 15])[:, None]
        output = p.propose_from_target(sampled_token_ids=[[20], [30]],
            aux_hidden_states=[positions.expand(-1, 2).to(torch.bfloat16)], num_decodes=2,
            prefill_data=None, decode_data=SimpleNamespace(position_ids=positions),
            scheduler_output=SimpleNamespace(num_scheduled_tokens={'a': 2, 'b': 2}))
        self.assertEqual(p.format_proposals(['a', 'b'], output.tolist()), [[6], [16, 17]])
        self.assertEqual(p.proposed_tokens, 3)

    def test_disabled_request_does_not_project_context_or_run_drafter(self):
        p = self.proposer(('a',))
        p.runner.requests['a'].sampling_params.extra_args = {'draft_token_limit': 0}
        p.observe_prefill(['a'], torch.arange(4)[None], None, [4], merged=False)
        output = p.propose_from_target(sampled_token_ids=[[20]], aux_hidden_states=[torch.zeros(1, 2)],
            num_decodes=1, prefill_data=None, decode_data=SimpleNamespace(position_ids=torch.tensor([[4]])),
            scheduler_output=SimpleNamespace(num_scheduled_tokens={'a': 1}))
        self.assertEqual(output.tolist(), [[-1, -1]])
        self.assertEqual(p.draft.projected, [])
        self.assertEqual(p.context.owners, {})
        self.assertEqual(p.calls, 0)

    def test_zero_alloc_meta_buffer_reuse_and_identity_preservation(self):
        p = self.proposer(('a', 'b'))
        seen_args = []
        orig_forward = p.model.forward

        def spy_forward(features, query_positions, counts, slots, anchors, next_positions, proposal_count=None):
            seen_args.append((features, query_positions, counts, slots, anchors, next_positions))
            return orig_forward(features, query_positions, counts, slots, anchors, next_positions, proposal_count)

        p.model.forward = spy_forward
        pos_tensor = torch.tensor([[4, 5], [14, 15]], dtype=torch.int32)
        feat_tensor = pos_tensor.reshape(-1, 1).expand(-1, 2).contiguous().to(torch.bfloat16)
        decode = SimpleNamespace(position_ids=pos_tensor)
        sched = SimpleNamespace(num_scheduled_tokens={'a': 2, 'b': 2})

        out1 = p.propose_from_target(sampled_token_ids=[[20], [30, 31]], aux_hidden_states=[feat_tensor],
                                     num_decodes=2, prefill_data=None, decode_data=decode, scheduler_output=sched)
        out2 = p.propose_from_target(sampled_token_ids=[[21, 22], [32]], aux_hidden_states=[feat_tensor],
                                     num_decodes=2, prefill_data=None, decode_data=decode, scheduler_output=sched)
        self.assertEqual(out1.tolist(), [[6, 7], [17, 18]])
        self.assertEqual(out2.tolist(), [[7, 8], [16, 17]])
        self.assertEqual(len(seen_args), 2)
        # Verify exact tensor object identity is preserved across steps for HPU graph copy_to skip_identity
        for idx in range(6):
            self.assertIs(seen_args[0][idx], seen_args[1][idx])
        self.assertIs(seen_args[0][0], feat_tensor)
        self.assertIs(seen_args[0][1], pos_tensor)

    def test_multilayer_aux_features_and_padded_decode_slicing(self):
        draft = FakeDraft()
        draft.spec = SimpleNamespace(layers=2, window=4, kv_heads=1, dim=2, hidden=2, target_layers=(0, 2, 4), block=4)

        def multi_project(features, positions):
            draft.projected.append((features.clone(), positions.clone()))
            k0 = features[:, :, :2].unsqueeze(2)
            v0 = features[:, :, 2:4].unsqueeze(2)
            k1 = features[:, :, 4:6].unsqueeze(2)
            v1 = (features[:, :, :2] + 1).unsqueeze(2)
            return ((k0, v0), (k1, v1))

        draft.project_context = multi_project
        p = self.proposer(('a', 'b'))
        p.draft = draft
        p.context = ContextPool(draft, 3, 'cpu')
        p.model = ProposalProgram(draft, p.context, p.target, p.num_speculative_tokens)

        # Test observe_prefill with 3 target layers
        prefill_pos = torch.tensor([[0, 1, 2]])
        aux_prefill = [torch.full((3, 2), float(layer + 1), dtype=torch.bfloat16) for layer in range(3)]
        p.observe_prefill(['a'], prefill_pos, aux_prefill, [3], merged=False)
        pairs, pos, valid = self.state(p, ['a'])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pos[0][valid[0]].tolist(), [0, 1, 2])
        self.assertEqual(pairs[0][0][0, :3, 0, 0].float().tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(pairs[0][1][0, :3, 0, 0].float().tolist(), [2.0, 2.0, 2.0])
        self.assertEqual(pairs[1][0][0, :3, 0, 0].tolist(), [3.0, 3.0, 3.0])

        # Test propose_from_target with padded HPU decode tensors (padded to 6 rows instead of 4 active rows)
        padded_pos = torch.tensor([4, 5, 14, 15, 99, 99], dtype=torch.int32)[:, None]
        aux_decode = [
            (padded_pos.to(torch.bfloat16) + layer * 10).expand(-1, 2).contiguous()
            for layer in range(3)
        ]
        out = p.propose_from_target(
            sampled_token_ids=[[20, 21], [30]],
            aux_hidden_states=aux_decode,
            num_decodes=2,
            prefill_data=None,
            decode_data=SimpleNamespace(position_ids=padded_pos),
            scheduler_output=SimpleNamespace(num_scheduled_tokens={'a': 2, 'b': 2}),
        )
        self.assertEqual(out.tolist(), [[7, 8], [16, 17]])

    def test_context_pool_unified_storage_views_and_apply(self):
        draft = FakeDraft()
        draft.spec = SimpleNamespace(layers=2, window=4, kv_heads=1, dim=2, hidden=2, target_layers=(0,), block=4)
        pool = ContextPool(draft, 3, 'cpu')
        slot_a = pool.slot('a')
        pos = torch.tensor([[0, 1]], dtype=torch.int32)
        k0 = torch.full((1, 2, 1, 2), 3.0, dtype=torch.bfloat16)
        v0 = torch.full((1, 2, 1, 2), 4.0, dtype=torch.bfloat16)
        k1 = torch.full((1, 2, 1, 2), 5.0, dtype=torch.bfloat16)
        v1 = torch.full((1, 2, 1, 2), 6.0, dtype=torch.bfloat16)
        pool.commit_slots(((k0, v0), (k1, v1)), pos, torch.tensor([2], dtype=torch.int32),
                          torch.tensor([slot_a], dtype=torch.int32))
        # Verify storage.key_i / value_i reflect unified _kv_all writes even after _apply
        pool = pool.to(dtype=torch.bfloat16)
        self.assertEqual(pool.storage.key_0[slot_a, 0, 0, 0].item(), 3.0)
        self.assertEqual(pool.storage.value_0[slot_a, 1, 0, 0].item(), 4.0)
        self.assertEqual(pool.storage.key_1[slot_a, 0, 0, 0].item(), 5.0)
        self.assertEqual(pool.storage.value_1[slot_a, 1, 0, 0].item(), 6.0)

    def test_proposal_program_uses_shard_local_greedy_reduction_without_full_logits_gather(self):
        p = self.proposer(('a', 'b'))
        # Replace p.target with a shard-local lm_head + logits_processor target whose
        # compute_logits raises if called, proving ProposalProgram uses compute_greedy_tokens.
        vocab_size = 32
        centers = torch.arange(vocab_size, dtype=torch.float32)

        def shard_apply(layer, hidden, bias=None):
            return -(centers[None] - hidden[:, :1].float()).abs()

        p.target = SimpleNamespace(
            embed_input_ids=lambda ids: ids[:, None].expand(-1, 2).to(torch.bfloat16),
            lm_head=SimpleNamespace(
                tp_rank=0,
                tp_size=1,
                num_embeddings_padded=vocab_size,
                num_embeddings_per_partition=vocab_size,
                org_vocab_size=vocab_size,
                bias=None,
                quant_method=SimpleNamespace(apply=shard_apply),
            ),
            logits_processor=SimpleNamespace(
                use_all_gather=True,
                logits_as_input=False,
                org_vocab_size=vocab_size,
                soft_cap=None,
                scale=1.0,
                head_dtype=None,
            ),
            compute_logits=lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError('ProposalProgram must use shard-local compute_greedy_tokens, not full compute_logits')
            ),
        )
        p.model = ProposalProgram(p.draft, p.context, p.target, p.num_speculative_tokens)
        self.assertIs(p.model.target_model, p.target)

        decode = SimpleNamespace(position_ids=torch.tensor([[4, 5, 6, 7], [14, 15, 16, 17]]))
        features = decode.position_ids.reshape(-1, 1).expand(-1, 2).to(torch.bfloat16)
        output = p.propose_from_target(
            sampled_token_ids=[[20], [30, 31, 32]],
            aux_hidden_states=[features],
            num_decodes=2,
            prefill_data=None,
            decode_data=decode,
            scheduler_output=SimpleNamespace(num_scheduled_tokens={'a': 4, 'b': 4}),
        )
        self.assertEqual(output.dtype, torch.int32)
        self.assertEqual(output.tolist(), [[6, 7], [18, 19]])


if __name__ == '__main__':
    unittest.main()

