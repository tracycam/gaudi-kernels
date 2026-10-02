"""Execute the production quantizer source with real HPU conversion.

AST extraction avoids importing an entire serving stack for this operator test.
It does not replace the implementation with a copy or mock the device cast.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc


def load_function(path):
    text = path.read_text()
    parsed = ast.parse(text)
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'dynamic_quant')
    scope = {'torch': torch, 'FP8_MAX': 240.0}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['dynamic_quant'], hashlib.sha256(text.encode()).hexdigest(), ast.get_source_segment(text, node)


def sync():
    hc.mark_step()
    torch.hpu.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-source', type=Path, required=True)
    p.add_argument('--candidate-source', type=Path, required=True)
    args = p.parse_args()
    out = Path(os.environ['PROBE_OUT'])
    torch.set_num_threads(4)
    torch.manual_seed(20260927)
    funcs, provenance = {}, {}
    for name, source in [('baseline', args.baseline_source), ('candidate', args.candidate_source)]:
        funcs[name], digest, text = load_function(source)
        provenance[name] = {'source': str(source), 'sha256': digest, 'function': text}
    records = []
    all_pass = True
    for m, k, single in [(1, 129, False), (1, 6144, False), (8, 6144, False),
                         (512, 6144, False), (8, 129, True)]:
        x0 = (torch.rand(m, k) * 2 - 1).bfloat16()
        x0[:, 0] = 1
        if m > 2:
            x0[1] = 0
            x0[2] *= 0.001
        amax = x0.float().abs().max() if single else x0.float().abs().amax(-1, keepdim=True)
        expected_scale = (amax + 1e-8) / 240
        expected_q = (x0.float() * expected_scale.reciprocal()).clamp(-240, 240).to(torch.float8_e4m3fn)
        x = x0.to('hpu')
        sync()
        variants, saved = {}, {}
        for name, fn in funcs.items():
            graph, stream = torch.hpu.HPUGraph(), torch.hpu.Stream()
            with torch.hpu.graph(graph, stream=stream):
                q, s = fn(x, single_scale=single)
            graph.replay()
            sync()
            q0, s0 = q.cpu(), s.cpu()
            saved[name] = {'q_bytes': q0.view(torch.uint8), 'scale': s0}
            for _ in range(10):
                graph.replay()
            sync()
            batches = []
            for _ in range(5):
                start = time.perf_counter_ns()
                for _ in range(50):
                    graph.replay()
                sync()
                batches.append((time.perf_counter_ns() - start) / 50 / 1000)
            variants[name] = {
                'scale_dtype': str(s0.dtype),
                'scale_word_mismatches_to_fp32': int((s0.view(torch.int32) != expected_scale.view(torch.int32)).sum()),
                'fp8_byte_mismatches_to_fp32': int((q0.view(torch.uint8) != expected_q.view(torch.uint8)).sum()),
                'relative_l2_to_input': float(((q0.float() * s0 - x0.float()).double().norm()
                                              / x0.double().norm().clamp_min(1e-30))),
                'synchronized_replay_us': statistics.median(batches),
                'all_batches_us': batches,
            }
            del graph
        record = {'M': m, 'K': k, 'single_scale': single, 'variants': variants,
                  'changed_bytes': int((saved['baseline']['q_bytes'] != saved['candidate']['q_bytes']).sum()),
                  'changed_scale_words': int((saved['baseline']['scale'].view(torch.int32)
                                              != saved['candidate']['scale'].view(torch.int32)).sum())}
        records.append(record)
        torch.save({'input': x0, 'expected_q': expected_q.view(torch.uint8),
                    'expected_scale': expected_scale, 'actual': saved}, out / f'M{m}-K{k}-single{int(single)}.pt')
        all_pass &= (variants['candidate']['scale_word_mismatches_to_fp32'] == 0
                     and variants['candidate']['fp8_byte_mismatches_to_fp32'] == 0)
        print(json.dumps(record), flush=True)
    result = {'status': 'PASS' if all_pass else 'FAIL', 'torch': torch.__version__,
              'scope': 'real HPU quantization bytes/scales and public replay wall; not model quality or device-only cycles',
              'provenance': provenance, 'records': records}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if not all_pass:
        raise AssertionError('HPU candidate disagrees with the specified FP32 quantization oracle')


if __name__ == '__main__':
    main()
