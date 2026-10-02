"""Independent old/new full-attention capture gate and ABBA timing/profile."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--extension', required=True)
p.add_argument('--profile-only', action='store_true')
p.add_argument('--replays', type=int, default=100)
a = p.parse_args()
out = Path(os.environ['PROBE_OUT'])
(out / 'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true', DUMP_POST_GRAPHS=str(out / 'post_graph.json'),
                  GRAPH_VISUALIZATION='1', GRAPH_VISUALIZATION_DIR=str(out / 'graphs'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'swa128_attention'))
from oracle import full_reference
torch.set_num_threads(4)
torch.manual_seed(281724)
torch.ops.load_library(a.extension)


def sync():
    hc.mark_step()
    torch.hpu.synchronize()


def bits(a, b):
    return int((a.view(torch.int32 if a.dtype == torch.float32 else torch.int16)
                != b.view(torch.int32 if b.dtype == torch.float32 else torch.int16)).sum())


def timing(graph, stream):
    events, walls = [], []
    for _ in range(3):
        start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
        with torch.hpu.stream(stream):
            begin = time.perf_counter(); start.record(stream)
            for _ in range(a.replays):
                graph.replay(asynchronous=True)
            end.record(stream); end.synchronize()
        events.append(start.elapsed_time(end) * 1000 / a.replays)
        walls.append((time.perf_counter() - begin) * 1e6 / a.replays)
    return {'event_us': events, 'wall_us': walls, 'median_event_us': statistics.median(events),
            'median_wall_us': statistics.median(walls),
            'scope': 'complete producer-attention-consumer graph; host replay supply may limit timing'}


# Structural Meta gate runs on the exact loaded 2.11 bridge.
mq = torch.empty((1, 1, 3072), device='meta', dtype=torch.bfloat16)
mk = torch.empty((32768, 1, 192), device='meta', dtype=torch.bfloat16)
mv = torch.empty((32768, 1, 128), device='meta', dtype=torch.bfloat16)
mp = torch.empty((2,), device='meta', dtype=torch.long)
mpos = torch.empty((1, 1), device='meta', dtype=torch.long)
msink = torch.empty((16,), device='meta', dtype=torch.bfloat16)
for name in ['baseline', 'quad', 'baseline_scores', 'quad_scores']:
    result = getattr(torch.ops.gaudi_swa128_ilp, name)(mq, mk, mv, mp, mp, mpos, msink, 192**-.5)
    tensors = result if isinstance(result, tuple) else (result,)
    assert all(t.shape == (1, 16, 128) for t in tensors)
    assert tensors[0].dtype == torch.bfloat16
    if len(tensors) == 2:
        assert tensors[1].dtype == torch.float32

q0 = torch.randn(1, 1, 3072).bfloat16()
k0 = torch.randn(32768, 1, 192).bfloat16()
v0 = torch.randn(32768, 1, 128).bfloat16()
sink0 = torch.tensor([-torch.inf, .5]).repeat(8).bfloat16()
offset0 = (torch.randn(1, 16, 128) * .01).bfloat16()
q, k, v, sink, offset = [x.to('hpu') for x in [q0, k0, v0, sink0, offset0]]
pages = torch.tensor([255, 0], dtype=torch.long, device='hpu')
groups = torch.tensor([0, 0], dtype=torch.long, device='hpu')
position = torch.tensor([[127]], dtype=torch.long, device='hpu')
sync()
torch.save(dict(query=q0, key=k0, value=v0, sinks=sink0, offset=offset0), out / 'inputs.pt')
graphs, streams, outputs, addresses = {}, {}, {}, {}
variants = ['baseline', 'quad'] if a.profile_only else ['baseline', 'quad', 'baseline_scores', 'quad_scores']
records, sequence = [], []
with torch.inference_mode():
    for name in variants:
        stream, graph = torch.hpu.Stream(), torch.hpu.HPUGraph()
        with torch.hpu.graph(graph, stream=stream):
            produced = q + .125
            result = getattr(torch.ops.gaudi_swa128_ilp, name)(produced, k, v, pages, groups, position, sink, 192**-.5)
            context, scores = result if name.endswith('scores') else (result, None)
            consumed = context + offset
        sync()
        graphs[name], streams[name], outputs[name] = graph, stream, (context, consumed, scores)
        addresses[name] = [t.data_ptr() for t in outputs[name] if t is not None]
        graph.replay(asynchronous=True); sync()
    fixtures = [(0, 'ordinary'), (1, 'ordinary'), (63, 'ordinary'), (64, 'ordinary'),
                (127, 'ordinary'), (128, 'ordinary'), (129, 'ordinary'), (320, 'ordinary'),
                (32766, 'ordinary'), (32767, 'ordinary'), (32704, 'cancellation'),
                (128, 'allmask'), (32767, 'maskedgroup'), (32767, 'sink')]
    if a.profile_only:
        fixtures = [(128, 'ordinary'), (32767, 'ordinary'), (0, 'ordinary')]
    for pos, kind in fixtures:
        qc, kc, sc = q0, k0, sink0
        pp, gg = ([255, 0] if pos % 2 == 0 else [0, 255]), [0, 0]
        if kind == 'cancellation':
            qc, kc = q0.clone(), k0.clone()
            qc[..., 0::2], qc[..., 1::2] = .875, -1.125  # after producer: +1/-1
            kc[..., 1::2] = kc[..., 0::2]
            kc[..., -1] += .0078125
        elif kind == 'allmask': pp = [-1, -1]
        elif kind == 'maskedgroup': gg = [-1, -1]
        elif kind == 'sink': sc = torch.full_like(sink0, 100)
        q.copy_(qc); k.copy_(kc); sink.copy_(sc)
        pages.copy_(torch.tensor(pp, dtype=torch.long)); groups.copy_(torch.tensor(gg, dtype=torch.long)); position.fill_(pos)
        sync()
        logical = max(0, pos - 127) // 128 * 128
        high, _ = full_reference((qc + .125).bfloat16(), kc, v0,
                                 [x if g == 0 else -1 for x, g in zip(pp, gg)], [logical, logical + 128], pos, sc)
        record = {'position': pos, 'kind': kind, 'pages': pp, 'groups': gg, 'variants': {}}
        raw = {'full_fp64': high, 'query': qc, 'sinks': sc, 'key_changed': kind == 'cancellation'}
        if kind == 'cancellation': raw['key'] = kc
        records.append(record)
        for name in variants:
            graphs[name].replay(asynchronous=True); sync()
            assert addresses[name] == [t.data_ptr() for t in outputs[name] if t is not None]
            ctx, consumed, score = [t.cpu() if t is not None else None for t in outputs[name]]
            raw[name] = {'context': ctx, 'consumed': consumed, 'scores': score}
            delta = ctx.double() - high
            e = {'finite': bool(ctx.isfinite().all()),
                 'relative_l2_fp64': float(delta.norm() / high.norm().clamp_min(1e-30)),
                 'max_abs_fp64': float(delta.abs().max()),
                 'context_baseline_bits': bits(ctx, raw['baseline']['context']),
                 'consumer_baseline_bits': bits(consumed, raw['baseline']['consumed'])}
            if name == 'quad_scores': e['scores_baseline_fp32_bits'] = bits(score, raw['baseline_scores']['scores'])
            record['variants'][name] = e
            torch.save(raw, out / f'p{pos}-{kind}.pt')
            (out / 'result.json').write_text(json.dumps({'status': 'RUNNING', 'records': records}, indent=2) + '\n')
            assert e['finite'] and e['context_baseline_bits'] == 0 and e['consumer_baseline_bits'] == 0, e
            assert e.get('scores_baseline_fp32_bits', 0) == 0, e
            assert e['relative_l2_fp64'] < .003, e
        if kind == 'ordinary' and pos in (0, 127, 128, 32767):
            for name in ['baseline', 'quad', 'quad', 'baseline']:
                sequence.append({'position': pos, 'variant': name, 'ordinal': len(sequence)})
                if a.profile_only:
                    for _ in range(3): graphs[name].replay(asynchronous=True)
                    sync()
                else:
                    sequence[-1]['timing'] = timing(graphs[name], streams[name])
        print(json.dumps(record), flush=True)
(out / 'result.json').write_text(json.dumps({'status': 'PASS_FULL_ATTENTION', 'records': records,
    'abba_sequence': sequence, 'profile_only': a.profile_only, 'torch': torch.__version__,
    'extension_sha256': hashlib.sha256(Path(a.extension).read_bytes()).hexdigest(),
    'scope': 'independent full-attention operator graph; no production or model TPS qualification'}, indent=2) + '\n')
