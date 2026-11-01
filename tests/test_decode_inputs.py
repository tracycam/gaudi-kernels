"""Compare prepared CPU metadata with the actual vLLM builder, without a model."""
import types
import unittest
import numpy as np
import torch

from gaudi_kernels.serving.decode_inputs import (
    PreparedDecodeInputs, eligible, prepare_single_query_values, prepare_spec_decode_inputs)
try:
    from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
except ModuleNotFoundError as exc:
    if exc.name != 'habana_frameworks':
        raise
    import ast
    import dataclasses
    import importlib.util
    import itertools
    import math
    from pathlib import Path
    import typing
    from torch.nn.utils.rnn import pad_sequence
    _u_src = Path(importlib.util.find_spec('vllm_gaudi.extension.utils').origin)
    _u_fns = [n for n in ast.parse(_u_src.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name in ('align_and_pad', 'pad_list')]
    _pa_src = Path(importlib.util.find_spec('vllm_gaudi.attention.ops.hpu_paged_attn').origin)
    _pa_cls = next(n for n in ast.parse(_pa_src.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'HPUPagedAttentionMetadata')
    _ba_src = Path(importlib.util.find_spec('vllm_gaudi.attention.backends.hpu_attn').origin)
    _ba_classes = [n for n in ast.parse(_ba_src.read_text()).body
                   if isinstance(n, ast.ClassDef) and 'Metadata' in n.name]
    _a_src = Path(importlib.util.find_spec('vllm_gaudi.v1.attention.backends.hpu_attn').origin)
    _a_cls = next(n for n in ast.parse(_a_src.read_text()).body
                  if isinstance(n, ast.ClassDef) and n.name == 'HPUAttentionMetadataV1')
    _r_src = Path(importlib.util.find_spec('vllm_gaudi.v1.worker.hpu_model_runner').origin)
    _r_tree = ast.parse(_r_src.read_text())
    _r_fns = [n for n in _r_tree.body if isinstance(n, ast.FunctionDef)
              and n.name in ('flatten', '_window_block_tables', 'gather_list')]
    _d_cls = next(n for n in _r_tree.body if isinstance(n, ast.ClassDef) and n.name == 'DecodeInputData')
    _cls = next(n for n in _r_tree.body if isinstance(n, ast.ClassDef) and n.name == 'HPUModelRunner')
    _wanted = {'_resolve_block', '_resolve_all_blocks', 'get_habana_paged_attn_buffers',
               '_create_decode_input_data', '_get_cumsum_and_arange', '_prepare_inputs',
               '_prepare_spec_decode_inputs'}
    _methods = [n for n in _cls.body if isinstance(n, ast.FunctionDef) and n.name in _wanted]

    @dataclasses.dataclass
    class _SpecDecodeMetadata:
        draft_token_ids: torch.Tensor
        num_draft_tokens: list[int]
        cu_num_draft_tokens: torch.Tensor
        cu_num_sampled_tokens: torch.Tensor
        target_logits_indices: torch.Tensor
        bonus_logits_indices: torch.Tensor
        logits_indices: torch.Tensor

    _mod = ast.fix_missing_locations(ast.Module(body=[
        *_u_fns, *_r_fns, _pa_cls, *_ba_classes, _a_cls, _d_cls,
        ast.ClassDef(name='HPUModelRunner', bases=[], keywords=[], body=_methods, decorator_list=[]),
    ], type_ignores=[]))
    _ns = {'torch': torch, 'np': np, 'math': math, 'itertools': itertools,
           'pad_sequence': pad_sequence,
           'AttentionMetadata': type('AttentionMetadata', (), {}),
           'SpecDecodeMetadata': _SpecDecodeMetadata, **vars(dataclasses), **vars(typing)}
    exec(compile(_mod, str(_r_src), 'exec'), _ns)
    HPUModelRunner = _ns['HPUModelRunner']


class Buckets:
    def find_decode_bucket(self, rows, pages, *args):
        return (1 << (rows-1).bit_length(), 1, max(8, ((pages+7)//8)*8))


def runner(count, window, contiguous):
    r = types.SimpleNamespace(
        attn_block_size=128, interleaved_sliding_window=window is not None,
        sliding_window=window, use_contiguous_pa=contiguous, _PAD_BLOCK_ID=1023,
        model_config=types.SimpleNamespace(dtype=torch.bfloat16), device='cpu',
        pin_memory=False, uses_mrope=False, model_has_chunked_attention=False,
        num_mamba_like_layers=0, use_async_scheduling=False, speculative_config=None,
        arange_np=np.arange(1024, dtype=np.int64),
        input_batch=types.SimpleNamespace(
            req_ids=[str(i) for i in range(count)],
            req_id_to_index={str(i): i for i in range(count)}),
        _step_decode_count=count, bucketing_manager=Buckets(),
        vllm_config=types.SimpleNamespace(parallel_config=types.SimpleNamespace(
            data_parallel_size=1, pipeline_parallel_size=1)),
        get_dp_padding=lambda n: 0, transfers={}, destinations={}, uploads=[])
    for name in ('_resolve_block', '_resolve_all_blocks', 'get_habana_paged_attn_buffers',
                 '_get_cumsum_and_arange', '_prepare_spec_decode_inputs'):
        setattr(r, name, types.MethodType(getattr(HPUModelRunner, name), r))

    def copy(role, source, **kwargs):
        key = (role, tuple(source.shape), source.dtype)
        if key not in r.destinations:
            r.destinations[key] = torch.empty_like(source)
        dst = r.destinations[key]
        dst.copy_(source)
        r.transfers[role] = (source, dst)
        r.uploads.append(role)
        return dst
    r.copy_input_to_device = copy
    return r


def values(r, positions, step, scheduled=None):
    if scheduled is None:
        r.positions_cpu = torch.tensor(positions, dtype=torch.int32)
        r.input_ids_cpu = torch.arange(len(positions), dtype=torch.int32) + 100 + step
        return
    expanded = [int(p) + j for p, n in zip(positions, scheduled) for j in range(int(n))]
    r.positions_cpu = torch.tensor(expanded, dtype=torch.int32)
    r.input_ids_cpu = torch.arange(len(expanded), dtype=torch.int32) + 100 + step * 17


def build(r, positions, pages, scheduled=None, scheduler_output=None):
    r.transfers = {}
    counts = [1] * len(positions) if scheduled is None else list(scheduled)
    result = HPUModelRunner._create_decode_input_data(
        r, len(positions), counts, np.asarray(positions), pages, scheduler_output)
    return result, dict(r.transfers)


class DecodeInputsTest(unittest.TestCase):
    def test_unit_query_staging_matches_original_ragged_preparation(self):
        for count in (1,2,3,8):
            target, reference = runner(count,128,False), runner(count,128,False)
            for r in (target, reference):
                r.arange_np = np.arange(128,dtype=np.int64)
                r.positions_cpu = torch.full((128,),-1,dtype=torch.int64)
                r.positions_np = r.positions_cpu.numpy()
                r.input_ids_cpu = torch.full((128,),-1,dtype=torch.int32)
                r.input_batch.token_ids_cpu_tensor = torch.arange(count*33000,dtype=torch.int32).reshape(count,33000)
                r.input_batch.token_ids_cpu = r.input_batch.token_ids_cpu_tensor.numpy()
                r.input_batch.num_prompt_tokens = [1]*count
                r._get_cumsum_and_arange = types.MethodType(HPUModelRunner._get_cumsum_and_arange,r)
                r._prepare_prefill_inputs = lambda *args: None
                r._prepare_decode_inputs = lambda *args: None
            scheduler = types.SimpleNamespace(total_num_scheduled_tokens=count,
                num_scheduled_tokens={rid:1 for rid in target.input_batch.req_ids})
            for base in (0,127,128,32767,32768,129):
                for r in (target,reference):
                    r.input_batch.num_computed_tokens_cpu = np.arange(count,dtype=np.int64)+base
                self.assertEqual(prepare_single_query_values(target,scheduler,count),[1]*count)
                HPUModelRunner._prepare_inputs(reference,scheduler,0,count)
                torch.testing.assert_close(target.positions_cpu,reference.positions_cpu,rtol=0,atol=0)
                torch.testing.assert_close(target.input_ids_cpu,reference.input_ids_cpu,rtol=0,atol=0)
            before = target.input_ids_cpu.clone()
            scheduler.num_scheduled_tokens['0'] = 2
            self.assertIsNone(prepare_single_query_values(target,scheduler,count))
            torch.testing.assert_close(before,target.input_ids_cpu,rtol=0,atol=0)

    def test_multi_query_speculative_staging_matches_original_ragged_preparation(self):
        for count, scheduled in ((1, [8]), (2, [8, 8]), (3, [8, 1, 5])):
            target, reference = runner(count, 128, False), runner(count, 128, False)
            for r in (target, reference):
                r.speculative_config = types.SimpleNamespace(num_speculative_tokens=7)
                r.arange_np = np.arange(128, dtype=np.int64)
                r.positions_cpu = torch.full((128,), -1, dtype=torch.int64)
                r.positions_np = r.positions_cpu.numpy()
                r.input_ids_cpu = torch.full((128,), -1, dtype=torch.int32)
                r.input_batch.token_ids_cpu_tensor = torch.arange(count * 4096, dtype=torch.int32).reshape(count, 4096)
                r.input_batch.token_ids_cpu = r.input_batch.token_ids_cpu_tensor.numpy()
                r.input_batch.num_prompt_tokens = [1] * count
                r._prepare_prefill_inputs = lambda *args: None
                r._prepare_decode_inputs = lambda *args: None
            scheduler = types.SimpleNamespace(
                total_num_scheduled_tokens=sum(scheduled),
                num_scheduled_tokens={str(i): scheduled[i] for i in range(count)})
            for base in (0, 64, 121, 125, 128, 1024, 1020):
                for r in (target, reference):
                    r.input_batch.num_computed_tokens_cpu = np.arange(count, dtype=np.int64) * 11 + base
                self.assertEqual(prepare_single_query_values(target, scheduler, count), scheduled)
                HPUModelRunner._prepare_inputs(reference, scheduler, 0, count)
                torch.testing.assert_close(target.positions_cpu, reference.positions_cpu, rtol=0, atol=0)
                torch.testing.assert_close(target.input_ids_cpu, reference.input_ids_cpu, rtol=0, atol=0)

    def test_32k_layout_updates_match_original_in_both_usage_dtypes(self):
        for dtype in (torch.bfloat16, torch.float32):
            target, reference = runner(3,128,False), runner(3,128,False)
            target.model_config.dtype = reference.model_config.dtype = dtype
            pages = torch.randperm(900, generator=torch.Generator().manual_seed(19)).to(torch.int32).reshape(3,300)
            cache = None
            for step, base in enumerate((32765,32766,32767,32768,32769,32770)):
                positions = [base,base-512,base-1024]
                values(target,positions,step); values(reference,positions,step)
                result = cache.update(target,3,positions,pages) if cache else None
                if result is None:
                    result, transfers = build(target,positions,pages)
                    cache = PreparedDecodeInputs(target,3,positions,pages,result,transfers)
                _, expected = build(reference,positions,pages)
                for role, (_, actual) in cache.transfers.items():
                    torch.testing.assert_close(actual,expected[role][1],rtol=0,atol=0)

    def test_all_roles_match_original_builder_across_layout_boundaries(self):
        hits = 0
        for count in (1, 2, 3, 8):
            for window in (None, 64, 128, 256):
                for contiguous in (False, True):
                    target, reference = runner(count, window, contiguous), runner(count, window, contiguous)
                    rng = np.random.default_rng(12)
                    pages = torch.from_numpy(rng.choice(1000, size=(count,80), replace=False).astype(np.int32))
                    cache = None
                    sequence = (0,1,2,63,64,65,126,127,128,129,254,255,256,257,4095,4096,4097,4096)
                    for step, base in enumerate(sequence):
                        positions = [base+i*3 for i in range(count)]
                        values(target, positions, step); values(reference, positions, step)
                        result = cache.update(target, count, positions, pages) if cache else None
                        if result is None:
                            result, transfers = build(target, positions, pages)
                            cache = PreparedDecodeInputs(target, count, positions, pages, result, transfers)
                        else:
                            hits += 1
                        _, expected = build(reference, positions, pages)
                        for role, (_, actual) in cache.transfers.items():
                            with self.subTest(count=count, window=window, contiguous=contiguous,
                                              position=base, role=role):
                                torch.testing.assert_close(actual, expected[role][1], rtol=0, atol=0)
        self.assertGreater(hits, 100)

    def test_multi_query_speculative_decode_matches_original_across_boundaries(self):
        hits = 0
        for count, scheduled in ((1, [8]), (2, [8, 8]), (3, [8, 4, 1])):
            for window in (None, 128, 256):
                for contiguous in (False, True):
                    target, reference = runner(count, window, contiguous), runner(count, window, contiguous)
                    for r in (target, reference):
                        r.speculative_config = types.SimpleNamespace(num_speculative_tokens=7)
                    orig_spec_target = target._prepare_spec_decode_inputs
                    target._prepare_spec_decode_inputs = (
                        lambda so, li, tid, m, _r=target, _orig=orig_spec_target:
                        prepare_spec_decode_inputs(_r, _orig, so, li, tid, m))
                    rng = np.random.default_rng(29)
                    pages = torch.from_numpy(rng.choice(1000, size=(count, 80), replace=False).astype(np.int32))
                    cache = None
                    sequence = (16, 20, 28, 36, 118, 124, 130, 136, 144, 248, 253, 260, 268, 265)
                    for step, base in enumerate(sequence):
                        positions = [base + i * 5 for i in range(count)]
                        scheduler_output = types.SimpleNamespace(
                            scheduled_spec_decode_tokens={
                                str(i): [1000 + step * 10 + i * 3 + j for j in range(scheduled[i] - 1)]
                                for i in range(count) if scheduled[i] > 1
                            })
                        values(target, positions, step, scheduled)
                        values(reference, positions, step, scheduled)
                        result = (cache.update(target, count, positions, pages, scheduled, scheduler_output)
                                  if cache else None)
                        if result is None:
                            result, transfers = build(target, positions, pages, scheduled, scheduler_output)
                            cache = PreparedDecodeInputs(
                                target, count, positions, pages, result, transfers, scheduled, scheduler_output)
                        else:
                            hits += 1
                        expected_result, expected = build(reference, positions, pages, scheduled, scheduler_output)
                        for role, (_, actual) in cache.transfers.items():
                            with self.subTest(count=count, scheduled=scheduled, window=window,
                                              contiguous=contiguous, position=base, role=role):
                                torch.testing.assert_close(actual, expected[role][1], rtol=0, atol=0)
                        spec_actual = result.spec_decode_metadata
                        spec_expected = expected_result.spec_decode_metadata
                        self.assertIsNotNone(spec_actual)
                        self.assertEqual(spec_actual.num_draft_tokens, spec_expected.num_draft_tokens)
                        for field in ('draft_token_ids', 'cu_num_draft_tokens', 'cu_num_sampled_tokens',
                                      'target_logits_indices', 'bonus_logits_indices', 'logits_indices'):
                            torch.testing.assert_close(
                                getattr(spec_actual, field), getattr(spec_expected, field), rtol=0, atol=0)
        self.assertGreater(hits, 60)

    def test_page_mutation_and_request_reorder_invalidate_before_upload(self):
        r = runner(2, 128, False)
        pages = torch.arange(160, dtype=torch.int32).reshape(2,80)
        positions = [4096,4097]
        values(r, positions, 0)
        result, transfers = build(r, positions, pages)
        cache = PreparedDecodeInputs(r, 2, positions, pages, result, transfers)
        calls = len(r.uploads)
        changed = pages.clone(); changed[0,0] = 999
        self.assertIsNone(cache.update(r, 2, positions, changed))
        r.input_batch.req_ids.reverse()
        self.assertIsNone(cache.update(r, 2, positions, pages))
        self.assertEqual(len(r.uploads), calls)

    def test_prepare_spec_decode_inputs_reuses_buffers_across_page_boundaries_and_invalidates_on_count_change(self):
        target, reference = runner(1, None, False), runner(1, None, False)
        for r in (target, reference):
            r.speculative_config = types.SimpleNamespace(num_speculative_tokens=7)
        orig_spec_target = target._prepare_spec_decode_inputs
        target._prepare_spec_decode_inputs = (
            lambda so, li, tid, m, _r=target, _orig=orig_spec_target:
            prepare_spec_decode_inputs(_r, _orig, so, li, tid, m))
        pages = torch.arange(80, dtype=torch.int32).reshape(1, 80)
        scheduled = [8]
        scheduler_output = types.SimpleNamespace(scheduled_spec_decode_tokens={'0': [1, 2, 3, 4, 5, 6, 7]})
        values(target, [120], 0, scheduled)
        res0, tr0 = build(target, [120], pages, scheduled, scheduler_output)
        cache = PreparedDecodeInputs(target, 1, [120], pages, res0, tr0, scheduled, scheduler_output)
        meta0 = res0.spec_decode_metadata
        # Cross 128-token block boundary -> PreparedDecodeInputs misses, but prepare_spec_decode_inputs hits!
        values(target, [130], 1, scheduled)
        values(reference, [130], 1, scheduled)
        self.assertIsNone(cache.update(target, 1, [130], pages, scheduled, scheduler_output))
        res1, _ = build(target, [130], pages, scheduled, scheduler_output)
        exp1, _ = build(reference, [130], pages, scheduled, scheduler_output)
        meta1 = res1.spec_decode_metadata
        self.assertIs(meta1.cu_num_draft_tokens, meta0.cu_num_draft_tokens)
        self.assertIs(meta1.cu_num_sampled_tokens, meta0.cu_num_sampled_tokens)
        self.assertIs(meta1.target_logits_indices, meta0.target_logits_indices)
        self.assertIs(meta1.bonus_logits_indices, meta0.bonus_logits_indices)
        self.assertIs(meta1.draft_token_ids, meta0.draft_token_ids)
        torch.testing.assert_close(meta1.draft_token_ids, exp1.spec_decode_metadata.draft_token_ids, rtol=0, atol=0)
        # Change draft token count -> invalidates cache
        scheduler_output_short = types.SimpleNamespace(scheduled_spec_decode_tokens={'0': [1, 2, 3]})
        values(target, [138], 2, [4])
        values(reference, [138], 2, [4])
        res2, _ = build(target, [138], pages, [4], scheduler_output_short)
        exp2, _ = build(reference, [138], pages, [4], scheduler_output_short)
        self.assertEqual(res2.spec_decode_metadata.num_draft_tokens, [3])
        torch.testing.assert_close(
            res2.spec_decode_metadata.draft_token_ids, exp2.spec_decode_metadata.draft_token_ids, rtol=0, atol=0)

    def test_unsupported_modes_use_original_builder(self):
        r = runner(2,128,False)
        self.assertTrue(eligible(r,2,[1,1]))
        self.assertFalse(eligible(r,2,[1,2]))
        r.speculative_config = object()
        self.assertFalse(eligible(r,2,[1,1]))
        r.speculative_config = types.SimpleNamespace(num_speculative_tokens=7)
        self.assertTrue(eligible(r,2,[1,8]))
        self.assertFalse(eligible(r,2,[0,8]))
        r.speculative_config = None
        r.defragmenter = types.SimpleNamespace(enabled=False)
        self.assertTrue(eligible(r,2,[1,1]))
        r.defragmenter.enabled = True
        self.assertFalse(eligible(r,2,[1,1]))
        r.defragmenter = None
        r._input_audit = types.SimpleNamespace(audit_inputs={})
        self.assertFalse(eligible(r,2,[1,1]))
        r._input_audit = None
        r.use_async_scheduling = True
        self.assertFalse(eligible(r,2,[1,1]))


if __name__ == '__main__':
    unittest.main()
