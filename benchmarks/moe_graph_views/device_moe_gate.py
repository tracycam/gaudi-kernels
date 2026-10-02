"""Real executor compact/broadcast graph clone ablation on one locked card.

All expert kernels and numeric casts are the frozen executor's original ones.
Only the three initial clones and separate precision-combine clone are gated.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import types

p = argparse.ArgumentParser()
p.add_argument('--executor-fixture', type=Path, required=True)
p.add_argument('--helper', type=Path, required=True)
p.add_argument('--helper-sha256', required=True)
p.add_argument('--quick', action='store_true')
p.add_argument('--mode-pair', action='store_true', help='same live M1 inputs/weights: compact versus broadcast; no clone rewrite')
p.add_argument('--constant-one-diagnostic', type=Path, help='separate experimental Torch library; compares original GP to literal-one GP')
p.add_argument('--grouped-activation-gp', type=Path, help='general K32 tensor fetch/broadcast GP candidate')
p.add_argument('--grouped-activation-down', type=Path, help='general K32 tensor fetch/broadcast down candidate')
p.add_argument('--fixed-grouped-gp', type=Path, help='keep the already-qualified grouped GP in both down variants')
p.add_argument('--folded-manifest', type=Path, help='pinned same-process grouped control / GP fold / down fold / both')
a = p.parse_args()
if sum(bool(v) for v in (a.mode_pair,a.constant_one_diagnostic,a.grouped_activation_gp,a.grouped_activation_down,a.folded_manifest))>1:
    p.error('separate mode and constant-one experiments')
if a.fixed_grouped_gp and not a.grouped_activation_down:p.error('--fixed-grouped-gp requires down comparison')
root = Path(__file__).resolve().parents[2]
out = Path(os.environ['PROBE_OUT'])
if not os.environ.get('GAUDI_KERNELS_MODULE_ID') or os.environ.get('HABANA_VISIBLE_MODULES') != os.environ['GAUDI_KERNELS_MODULE_ID']:
    raise RuntimeError('single-module bounded runner required')
assert os.environ.get('PT_HPU_LAZY_MODE') == '1'
os.environ.update(PT_ENABLE_INT64_SUPPORT='0', UNIFIED_PRECISION_ROUTER='1', NATIVE_BRIDGE='pt2',
                  ENABLE_EXPERIMENTAL_FLAGS='true', DUMP_POST_GRAPHS=str(out/'post_graph.json'),
                  GRAPH_VISUALIZATION='1', GRAPH_VISUALIZATION_DIR=str(out/'graphs'))
(out/'graphs').mkdir()
fixture = a.executor_fixture.resolve(strict=True)
snapshot = json.loads((fixture/'snapshot.json').read_text())
for name, sha in snapshot['files_sha256'].items():
    assert hashlib.sha256((fixture/name).read_bytes()).hexdigest() == sha, name
(out/'executor-snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n')
folded_plan=None
if a.folded_manifest:
    sys.path.insert(0,str(root/'tools'))
    from moe_folded_plan import validate
    folded_plan=validate(a.folded_manifest)
    (out/'folded-plan.json').write_text(json.dumps(folded_plan,indent=2)+'\n')
import numpy as np
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
torch.set_num_threads(2)
sys.path.insert(0, str(root/'python'))
sys.path.insert(0, str(fixture/'executor'))
from gaudi_kernels import moe_private_boundary as boundary
pin = boundary.install(a.helper, a.helper_sha256)
import native_ops
from kernels.packing import pack, unpack

def module_from_source(name, path, rewrite):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    source = path.read_text()
    rewritten = rewrite(source, module.__dict__)
    exec(compile(rewritten, str(path), 'exec'), module.__dict__)
    (out/(name+'-executed.py')).write_text(rewritten)
    return module

precision = module_from_source('precision_ops', fixture/'executor/precision_ops.py', boundary.rewrite_precision_source)
batch = module_from_source('batch_ops', fixture/'executor/batch_ops.py', boundary.rewrite_batch_source)
if a.constant_one_diagnostic or a.grouped_activation_gp or a.fixed_grouped_gp:
    torch.ops.load_library(str((a.constant_one_diagnostic or a.grouped_activation_gp or a.fixed_grouped_gp).resolve(strict=True)))
    original_gp = torch.ops.unified_batch.gp
    def diagnostic_gp(*args):
        flag=os.environ.get('GK_DIAGNOSTIC_GP')
        if flag=='broadcast' or a.fixed_grouped_gp:return torch.ops.gaudi_gp_diagnostic.broadcast(*args)
        return torch.ops.gaudi_gp_diagnostic.immediate_one(*args) if flag=='1' else original_gp(*args)
    # Local fixture module only. Original operator and all other callers stay unchanged.
    source = (out/'batch_ops-executed.py').read_text()
    needle = 'gp_out=torch.ops.unified_batch.gp(gp,gs,x,table,ids)'
    assert source.count(needle) == 1
    source = source.replace(needle, 'gp_out=_diagnostic_gp(gp,gs,x,table,ids)')
    batch.__dict__['_diagnostic_gp'] = diagnostic_gp
    function = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'moe']
    assert len(function) == 1
    exec(compile(ast.Module(body=function, type_ignores=[]), str(fixture/'executor/batch_ops.py'), 'exec'), batch.__dict__)
    (out/'batch_ops-diagnostic-executed.py').write_text(source)
if a.grouped_activation_down:
    torch.ops.load_library(str(a.grouped_activation_down.resolve(strict=True)))
    original_down=torch.ops.unified_batch.down
    def diagnostic_down(*args):
        return torch.ops.gaudi_down_activation.broadcast(*args) if os.environ.get('GK_DIAGNOSTIC_DOWN')=='1' else original_down(*args)
    source=(out/('batch_ops-diagnostic-executed.py' if a.fixed_grouped_gp else 'batch_ops-executed.py')).read_text()
    needle='partial=torch.ops.unified_batch.down(dp,ds,gate,table,ids)';assert source.count(needle)==1
    source=source.replace(needle,'partial=_diagnostic_down(dp,ds,gate,table,ids)')
    batch.__dict__['_diagnostic_down']=diagnostic_down
    function=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='moe'];assert len(function)==1
    exec(compile(ast.Module(body=function,type_ignores=[]),str(fixture/'executor/batch_ops.py'),'exec'),batch.__dict__)
    (out/'batch_ops-down-executed.py').write_text(source)
if folded_plan:
    for key in ('gp_torch','down_torch','folded_torch'):
        torch.ops.load_library(folded_plan['files'][key]['path'])
    def folded_gp(*args):
        return (torch.ops.gaudi_activation_folded.gp if capture_variant in ('folded_gp_only','folded_both')
                else torch.ops.gaudi_gp_diagnostic.broadcast)(*args)
    def folded_down(*args):
        return (torch.ops.gaudi_activation_folded.down if capture_variant in ('folded_down_only','folded_both')
                else torch.ops.gaudi_down_activation.broadcast)(*args)
    source=(out/'batch_ops-executed.py').read_text()
    for old,new in [('gp_out=torch.ops.unified_batch.gp(gp,gs,x,table,ids)','gp_out=_folded_gp(gp,gs,x,table,ids)'),
                    ('partial=torch.ops.unified_batch.down(dp,ds,gate,table,ids)','partial=_folded_down(dp,ds,gate,table,ids)')]:
        assert source.count(old)==1;source=source.replace(old,new)
    batch.__dict__.update(_folded_gp=folded_gp,_folded_down=folded_down)
    function=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='moe'];assert len(function)==1
    exec(compile(ast.Module(body=function,type_ignores=[]),str(fixture/'executor/batch_ops.py'),'exec'),batch.__dict__)
    (out/'batch_ops-folded-executed.py').write_text(source)
def sync():
    hc.mark_step()
    torch.hpu.synchronize()
def same(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
def graph_ids():
    path = out/'post_graph.json'
    # Profiling switches the compiler from one aggregate file to a directory of
    # per-recipe JSON dumps. Preserve both layouts; never delete/recreate it.
    files = [path] if path.is_file() else list(path.rglob('*.json')) if path.is_dir() else []
    return sorted({g['id'] for file in files for g in json.loads(file.read_text())['graphs']})

result = {'status': 'STARTED', 'pin': pin, 'records': [], 'weights': {},
          'scope': 'single-rank executor clone ablation; original4bit storage, no model TPS claim',
          'numeric_gate': 'all final output bytes identical to unchanged clone baseline',
          'event_scope': 'capture-stream elapsed span including replay submission gaps; not summed engine active time'}
def save():
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')

try:
    rng = np.random.default_rng(927128)
    torch.manual_seed(927128)
    experts, topk = 24, 8
    cpu_weights = []
    for n, k in ((512, 6144), (6144, 256)):
        weights, scales = [], []
        for _ in range(experts):
            w, s = pack(rng.integers(0, 256, (n, k//2), dtype=np.uint8),
                        rng.integers(115, 124, (n, k//32), dtype=np.uint8))
            weights.append(w.reshape(-1, 256))
            scales.append(s.reshape(-1, 512))
        cpu_weights.extend([torch.from_numpy(np.concatenate(weights)), torch.from_numpy(np.concatenate(scales))])
    torch.save(dict(zip(('gp', 'gs', 'dp', 'ds'), cpu_weights)), out/'prepared-weights.pt')
    result['weights'] = {'expert_count': experts, 'local_gp_shape': [512, 6144], 'local_down_shape': [6144, 256],
                         'format': 'original E2M1/E8M0 per32, lossless N512 prepack',
                         'seed': 927128, 'source': 'bounded synthetic fixture, not model checkpoint',
                         'packed_payload_bytes': sum(w.numel()*w.element_size() for w in cpu_weights),
                         'prepared_sha256': hashlib.sha256((out/'prepared-weights.pt').read_bytes()).hexdigest()}
    save()
    gp, gs, dp, ds = [w.to('hpu') for w in cpu_weights]
    table, directions = native_ops.constants('hpu')
    sync()
    # A high-precision MAC reference for the small fixture, keeping the original
    # BF16 gate/up/SILU/product boundaries. GPU sigmoid approximation and FP32
    # dot ordering can differ; clone equivalence is separately byte-exact.
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
    dense_cache = {}
    def expert_weight(e, down):
        key = (e, down)
        if key not in dense_cache:
            n, k = (6144, 256) if down else (512, 6144)
            w, s = cpu_weights[2:] if down else cpu_weights[:2]
            blocks = n//512
            packed = w[e*blocks*k:(e+1)*blocks*k].numpy().reshape(blocks, k, 256)
            scale = s[e*blocks*(k//32):(e+1)*blocks*(k//32)].numpy().reshape(blocks, k//32, 512)
            original, scales = unpack(packed, scale)
            codes = np.empty((n, k), np.uint8)
            codes[:, ::2] = original & 15
            codes[:, 1::2] = original >> 4
            dense_cache[key] = lut[torch.from_numpy(codes).long()] * torch.pow(
                2., torch.from_numpy(scales).double()-127).repeat_interleave(32, dim=1)
        return dense_cache[key]
    def reference_one(x, logits, gain):
        values, ids = torch.topk(logits + .125, topk, dim=-1)
        routes = torch.softmax(values, -1, dtype=torch.float32) * gain
        activations = (x + torch.tensor(.015625, dtype=torch.bfloat16)).reshape(-1, 6144).double()
        result64 = torch.zeros(activations.shape[0], 6144, dtype=torch.float64)
        for row in range(activations.shape[0]):
            for slot, expert in enumerate(ids[row].tolist()):
                gu = expert_weight(expert, False) @ activations[row]
                gate, up = gu[:256].bfloat16(), gu[256:].bfloat16()
                hidden = (torch.nn.functional.silu(gate.float()).bfloat16()*up).bfloat16().double()
                result64[row] += (expert_weight(expert, True) @ hidden) * routes[row, slot].double()
                # The reference is only for M1. Do not accumulate hundreds of
                # MB of expanded CPU matrices across expert selections.
                dense_cache.clear()
        return result64 + .03125

    cases = [('compact', 1, False), ('compact', 1, True), ('broadcast', 8, False),
             ('broadcast', 8, True), ('broadcast', 64, True)]
    if a.quick:
        cases = cases[:2]
    variants = [('clone_baseline', '0', '0'), ('initial_three_removed', '1', '0'), ('all_four_removed', '1', '1')]
    if a.mode_pair:
        cases = [('paired', 1, False)]
        variants = [('compact', '0', '0'), ('broadcast', '0', '0')]
        result['scope'] = 'same-data M1 compact versus broadcast; original kernels/arithmetic; no clone rewrite or model TPS claim'
    if a.constant_one_diagnostic:
        cases = [('compact', 1, False)]
        variants = [('original_gp', '0', '0'), ('immediate_one_gp', '0', '0')]
        result['scope'] = 'constant BF16-one activation diagnostic; no general kernel, no default change'
    if a.grouped_activation_gp:
        cases = [('compact', 1, False),('compact',2,True),('compact',8,True),('compact',64,True)]
        if a.quick:cases=cases[:1]
        variants = [('original_gp','0','0'),('grouped_activation_gp','0','0')]
        result['scope']='general K32 grouped activation GP candidate, unchanged weight/MAC/scale order; no default change'
    if a.grouped_activation_down:
        cases=[('compact',1,False),('compact',2,True),('compact',8,True)]
        if a.quick:cases=cases[:1]
        variants=[('original_down','0','0'),('grouped_activation_down','0','0')]
        result['scope']='general K32 grouped down; original down MAC/scale and slot-combine order; no default change'
        result['fixed_grouped_gp']=bool(a.fixed_grouped_gp)
    if folded_plan:
        cases=[('compact',1,False),('compact',2,True),('compact',8,True)]
        if a.quick:cases=cases[:1]
        variants=[(v,'0','0') for v in ('grouped_control','folded_gp_only','folded_down_only','folded_both')]
        result['scope']='same-process folded GP/down ABBA; qualified grouped control, unchanged MAC/scale/combine order; no default change'
    with torch.inference_mode():
        for mode, rows, is3d in cases:
            name = f'{mode}-m{rows}-'+('3d' if is3d else '2d')
            dest = out/name
            dest.mkdir()
            shape = (1, rows, 6144) if is3d else (rows, 6144)
            x0 = (torch.randn(shape)*.1).bfloat16()
            if a.constant_one_diagnostic:
                x0.fill_(.984375)  # exact BF16 + .015625 producer gives exactly one
            logits0 = torch.randn(rows, experts)
            gain0 = torch.ones(rows, 1)
            raw = {'x': x0, 'logits': logits0, 'gain': gain0, 'mutations': []}
            torch.save(raw, dest/'fixture.pt')
            x, logits, gain = x0.to('hpu'), logits0.to('hpu'), gain0.to('hpu')
            xoffset = torch.tensor(.015625, dtype=torch.bfloat16).to('hpu')
            post = torch.tensor(.03125).to('hpu')
            sync()
            graphs, outputs, streams = [], [], []
            record = {'case': name, 'mode': mode, 'rows': rows, 'is3d': is3d, 'graphs': [], 'checks': [], 'timing': []}
            result['records'].append(record)
            def chain():
                produced = x + xoffset
                vals, ids = torch.topk(logits + .125, topk, dim=-1)
                route = torch.softmax(vals, -1, dtype=torch.float32) * gain
                if is3d:
                    ids = ids.reshape(1, rows, topk)
                    route = route.reshape(1, rows, topk)
                # Exactly the original plugin's temporary producer views.
                xx = produced.view(-1, 6144)
                ii = ids.view(-1, topk)
                rr = route.view(-1, topk)
                y = batch.moe(xx, ii, rr, gp, gs, dp, ds, table, directions, mode=current_mode)
                # Restore is consumed by a vendor op, not merely returned.
                return y.reshape(shape) + post
            for label, remove3, remove4 in variants:
                capture_variant=label
                current_mode = label if a.mode_pair else mode
                os.environ['GK_DIAGNOSTIC_GP'] = ('1' if label == 'immediate_one_gp' else
                                                'broadcast' if label == 'grouped_activation_gp' else '0')
                os.environ['GK_DIAGNOSTIC_DOWN']='1' if label=='grouped_activation_down' else '0'
                os.environ.update(GK_MOE_PRIVATE_VIEWS=remove3, GK_MOE_PRIVATE_COMBINE=remove4)
                before = set(graph_ids())
                stream, graph = torch.hpu.Stream(), torch.hpu.HPUGraph()
                with torch.hpu.graph(graph, stream=stream):
                    y = chain()
                sync()
                graphs.append(graph); outputs.append(y); streams.append(stream)
                record['graphs'].append({'variant': label, 'postgraph_ids': sorted(set(graph_ids())-before),
                                         'boundary': boundary.snapshot()})
                save()
            owners = [x.data_ptr(), logits.data_ptr(), gain.data_ptr(), *[y.data_ptr() for y in outputs]]
            hot = torch.full((rows, experts), -20.)
            hot[:, :topk] = torch.arange(topk, 0, -1).float()
            cold = torch.full((rows, experts), -20.)
            cold[:, -topk:] = torch.arange(topk, 0, -1).float()
            mutations = [('initial', x0, logits0, gain0),
                         ('changed_input_hot_experts', (x0.roll(17, -1)*1.125).bfloat16(), hot, gain0),
                         ('changed_cold_experts', -x0, cold, gain0),
                         ('empty_routes', x0, hot, torch.zeros_like(gain0))]
            if a.constant_one_diagnostic:
                mutations = [('initial', x0, logits0, gain0), ('changed_hot_experts', x0, hot, gain0),
                             ('changed_cold_experts', x0, cold, gain0), ('empty_routes', x0, hot, torch.zeros_like(gain0))]
            for mutation, xm, lm, gm in mutations:
                entry = {'name': mutation, 'x': xm, 'logits': lm, 'gain': gm}
                raw['mutations'].append(entry)
                torch.save(raw, dest/'fixture.pt')
                x.copy_(xm); logits.copy_(lm); gain.copy_(gm); sync()
                actual = []
                for graph, output in zip(graphs, outputs):
                    for _ in range(10): graph.replay(asynchronous=True)
                    sync()
                    # Read each variant before another graph can write output.
                    # Public graph-owned replay remains valid across readbacks;
                    # no physical launch binding is borrowed or replayed here.
                    actual.append(output.cpu())
                assert owners == [x.data_ptr(), logits.data_ptr(), gain.data_ptr(), *[y.data_ptr() for y in outputs]]
                entry['outputs'] = actual
                torch.save(raw, dest/'fixture.pt')
                checks = [same(actual[0], y) for y in actual[1:]]
                assert all(checks) and all(torch.isfinite(y).all() for y in actual)
                if mutation == 'empty_routes': assert torch.equal(actual[0], torch.full_like(actual[0], .03125))
                check = {'mutation': mutation, 'bitwise_variants_vs_baseline': checks,
                         'checked_values': sum(y.numel() for y in actual), 'same_live_logical_owner_handles': True,
                         'physical_address_claim': False, 'replays_per_variant': 10}
                if rows == 1:
                    reference = reference_one(xm, lm, gm)
                    entry['fp64_mac_reference'] = reference
                    diff = actual[0].reshape(rows, 6144).double()-reference
                    check['baseline_vs_fp64_mac_reference'] = {'relative_l2': float(diff.norm()/reference.norm().clamp_min(1e-30)),
                                                               'max_abs': float(diff.abs().max()),
                                                               'reference_scope': 'FP64 MAC with BF16 nonlinear boundaries; GPU approximate sigmoid may differ'}
                    torch.save(raw, dest/'fixture.pt')
                record['checks'].append(check); save()
            dense_cache.clear()
            x.copy_(x0); logits.copy_(logits0); gain.copy_(gain0); sync()
            for graph in graphs:
                for _ in range(3): graph.replay(asynchronous=True)
            sync()
            repeats = 16 if rows < 64 else 8
            for trial in range(3):
                order=([(candidate,index) for candidate in range(1,4) for index in (0,candidate,candidate,0)]
                       if folded_plan else [(None,index) for index in list(range(len(variants)))+list(reversed(range(len(variants))))])
                for paired_candidate,index in order:
                    stream, graph = streams[index], graphs[index]
                    begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                    sync(); started = time.perf_counter()
                    begin.record(stream)
                    for _ in range(repeats): graph.replay(asynchronous=True)
                    sync(); end.record(stream); end.synchronize(); sync()
                    record['timing'].append({'trial': trial, 'variant': variants[index][0], 'repeats': repeats,
                                             'paired_candidate': variants[paired_candidate][0] if paired_candidate else None,
                                             'event_us': begin.elapsed_time(end)*1000/repeats,
                                             'wall_us': (time.perf_counter()-started)*1e6/repeats})
            record['median_us'] = {label: {key: statistics.median(t[key] for t in record['timing'] if t['variant'] == label)
                                                  for key in ('event_us', 'wall_us')} for label, _, _ in variants}
            if folded_plan:
                record['pair_median_us']={candidate:{arm:{key:statistics.median(t[key] for t in record['timing']
                    if t['paired_candidate']==candidate and t['variant']==label) for key in ('event_us','wall_us')}
                    for arm,label in [('control','grouped_control'),('candidate',candidate)]} for candidate,_,_ in variants[1:]}
            record['state'] = 'PASS_BITWISE_AND_REPLAY'
            (dest/'result.json').write_text(json.dumps(record, indent=2)+'\n')
            save(); print(json.dumps({k: record[k] for k in ('case', 'state', 'median_us')}), flush=True)
            del graphs, outputs, streams, graph, y, x, logits, gain
            sync()
    result['status'] = 'PASS_BITWISE_AND_REPLAY'
except Exception as error:
    result.update(status='FAIL', error=repr(error))
    raise
finally:
    save()
print(json.dumps({'status': result['status'], 'cases': len(result['records'])}), flush=True)
