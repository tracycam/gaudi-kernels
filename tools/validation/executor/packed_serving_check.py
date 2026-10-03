"""Actual scheduler/allocator packed execution versus ordinary teacher logits."""
import json


def run(llm, root):
    import torch
    from vllm import SamplingParams
    from tools.validation.executor.batch_quality import generate_with_ids, attach_owners
    from gaudi_kernels.serving.executor.production_quality import compare_logits

    base = llm.get_tokenizer().encode('The quick brown fox jumps over the lazy dog. ')
    prompts = [{'prompt_token_ids': (base * 128)[:n]} for n in (125, 257, 769)]
    params = [SamplingParams(temperature=0, max_tokens=5, ignore_eos=True, allowed_token_ids=[101 + i])
              for i in range(3)]
    llm.collective_rpc('production_quality_capture', args=('scheduled-serving-control', 'batch', 1))
    baseline, mapping = generate_with_ids(llm, prompts, params)
    ranks = llm.collective_rpc('production_quality_capture', args=(None,))
    frames = attach_owners(next(rank['captured_forwards'] for rank in ranks if rank['rank'] == 0), baseline, mapping)
    references = {}
    for frame in frames:
        for row, (owner, query) in enumerate(zip(frame['request_indices'], frame['logits_indices'])):
            if owner is not None:
                key = (owner, frame['positions'][query], frame['input_ids'][query])
                if key in references:
                    raise ValueError('Duplicate ordinary serving teacher query')
                references[key] = (frame['path'], row)
    llm.collective_rpc('packed_scheduled_configure', args=(True,))
    try:
        outputs, mapping = generate_with_ids(llm, prompts, params)
    finally:
        candidate_ranks = llm.collective_rpc('packed_scheduled_configure', args=(False,))
    records = next(rank['records'] for rank in candidate_ranks if rank['rank'] == 0)
    indices = {output.request_id: index for index, output in enumerate(outputs)}
    candidates = {}
    for frame in records:
        for row, ((rid, _), query) in enumerate(zip(frame['output_owners'], frame['logits_indices'])):
            key = (indices[mapping[rid]], frame['positions'][query], frame['input_ids'][query])
            if key in candidates:
                raise ValueError('Duplicate packed scheduled teacher query')
            candidates[key] = (frame['path'], row)
    loaded = {}
    def load(location):
        path, row = location
        if path not in loaded:
            loaded[path] = torch.load(path, weights_only=True, map_location='cpu')['logits']
        return loaded[path][row:row+1]
    checks = [{'request': key[0], 'position': key[1], 'input_id': key[2],
               'check': compare_logits(load(references[key]), load(candidates[key]), contract='fp32_arithmetic_v1')}
              for key in sorted(references.keys() & candidates.keys())]
    coverage = all(sum(row['request'] == index for row in checks) >= 4 for index in range(3))
    mixed = any('decode' in frame['kinds'] and 'prefill' in frame['kinds'] for frame in records)
    no_output_chunks = any(len(frame['output_owners']) < len(frame['request_ids']) for frame in records)
    result = {'pass': coverage and mixed and no_output_chunks and all(row['check']['pass'] for row in checks),
              'checks': checks, 'all_request_coverage': coverage, 'actual_mixed_step': mixed,
              'incomplete_prefill_without_emission': no_output_chunks, 'ranks': candidate_ranks,
              'baseline_ids': [o.outputs[0].token_ids for o in baseline],
              'candidate_ids': [o.outputs[0].token_ids for o in outputs],
              'scope': 'actual scheduler and allocator; forced teacher input; two-layer diagnostics, no TPS'}
    (root/'packed-scheduled-result.json').write_text(json.dumps(result, indent=2)+'\n')
    if not result['pass']:
        raise RuntimeError('Actual packed scheduling teacher or mixed-step gate failed')
    return result
