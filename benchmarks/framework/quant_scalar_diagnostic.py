"""Isolate HPU FP32 division and scalar/ranked FP8 scale handling."""
import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc

out = Path(os.environ['PROBE_OUT'])
torch.manual_seed(20260928)
x0 = (torch.rand(8, 129) * 2 - 1).bfloat16()
x0[:, 0] = 1
x0[1] = 0
x0[2] *= .001
x = x0.to('hpu')
hc.mark_step()
torch.hpu.synchronize()
records = []
for reduction in ['row', 'scalar', 'scalar_reshape1', 'scalar_rank2', 'scalar_expand']:
    for method in ['div', 'mul']:
        for promote in [False, True]:
            graph = torch.hpu.HPUGraph()
            stream = torch.hpu.Stream()
            with torch.hpu.graph(graph, stream=stream):
                maximum = x.abs().amax(-1, keepdim=True).float() if reduction == 'row' else x.abs().max().float()
                if reduction == 'scalar_reshape1':
                    maximum = maximum.reshape(1)
                elif reduction == 'scalar_rank2':
                    maximum = maximum.reshape(1, 1)
                elif reduction == 'scalar_expand':
                    maximum = maximum.reshape(1, 1).expand(x.shape[0], 1)
                scale = (maximum + 1e-8) / 240 if method == 'div' else (maximum + 1e-8) * (1. / 240.)
                reciprocal = scale.reciprocal()
                value = x.float() if promote else x
                q = torch.ops.hpu.cast_to_fp8_v2(value, reciprocal, False, False, torch.float8_e4m3fn)[0]
            graph.replay()
            hc.mark_step()
            torch.hpu.synchronize()
            a, s, r, q0 = maximum.cpu(), scale.cpu(), reciprocal.cpu(), q.cpu()
            expected_a = x0.float().abs().amax(-1, keepdim=True) if reduction == 'row' else x0.float().abs().max()
            if reduction == 'scalar_reshape1':
                expected_a = expected_a.reshape(1)
            elif reduction == 'scalar_rank2':
                expected_a = expected_a.reshape(1, 1)
            elif reduction == 'scalar_expand':
                expected_a = expected_a.reshape(1, 1).expand(x.shape[0], 1)
            expected_s = (expected_a + 1e-8) / 240 if method == 'div' else (expected_a + 1e-8) * (1. / 240.)
            expected_r = expected_s.reciprocal()
            # Use the actually produced reciprocal to isolate the cast from its producer.
            expect_cast = (x0.float() * r).clamp(-240, 240).to(torch.float8_e4m3fn)
            want = (x0.float() * expected_r).clamp(-240, 240).to(torch.float8_e4m3fn)
            row = {'reduction': reduction, 'scale_method': method, 'promote_input': promote,
                   'maximum': a.flatten()[:3].tolist(), 'scale': s.flatten()[:3].tolist(),
                   'reciprocal': r.flatten()[:3].tolist(),
                   'scale_ulp': (s.view(torch.int32) - expected_s.view(torch.int32)).flatten().tolist(),
                   'reciprocal_ulp': (r.view(torch.int32) - expected_r.view(torch.int32)).flatten().tolist(),
                   'cast_mismatches_actual_reciprocal': int((q0.view(torch.uint8) != expect_cast.view(torch.uint8)).sum()),
                   'pipeline_mismatches': int((q0.view(torch.uint8) != want.view(torch.uint8)).sum())}
            records.append(row)
            print(json.dumps(row), flush=True)
            torch.save({'x': x0, 'amax': a, 'scale': s, 'reciprocal': r, 'q': q0.view(torch.uint8)},
                       out / f'{reduction}-{method}-f32input{int(promote)}.pt')
            del graph
(out / 'result.json').write_text(json.dumps({'status': 'DIAGNOSTIC', 'records': records}, indent=2) + '\n')
