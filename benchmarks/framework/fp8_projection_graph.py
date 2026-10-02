"""Pending-device experiment: shared-X projections, one captured graph per case.

Run only through run_device_probe.py after rebuilding/loading the public bridge.
Compare the same prepared weight bytes and whole graphs, including any output
materialization. --independent-baseline transfers separate CPU projection views
to independent HPU owners at loading time, avoiding unsupported input aliases.
Both baseline/fused weights coexist only for this A/B experiment; no weight copy
or expansion is added to a captured graph.
"""
import ctypes
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('--grid',choices=['smoke','main','tail','all'],default='all')
parser.add_argument('--warm-input-views',action='store_true')
parser.add_argument('--weight-views-inside',action='store_true')
parser.add_argument('--independent-baseline',action='store_true')
args=parser.parse_args()
assert not (args.independent_baseline and args.weight_views_inside)

out = Path(os.environ['PROBE_OUT'])
(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true', GRAPH_VISUALIZATION='1',
                  GRAPH_VISUALIZATION_DIR=str(out/'graphs'),
                  DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
from validation import timings_agree

root = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'python'))
from gaudi_kernels.torch_linear import load_extension, linear_fp8, PreparedSeparableFP8
from gaudi_kernels.fp8_projections import prepare_fp8_projections, linear_fp8_projections, split_fp8_projections
load_extension(root/os.environ['GK_TORCH_BUILD']/'gaudi_kernels_torch.so')
counter = ctypes.CDLL(None).gaudi_kernel_launch_count
counter.restype = ctypes.c_ulonglong
torch.set_num_threads(4)
torch.manual_seed(270929)


def sync():
    hc.mark_step()
    torch.hpu.synchronize()


def relative(actual, reference):
    return float((actual.double()-reference.double()).norm()/reference.double().norm().clamp_min(1e-30))


def projection_views(device,widths):
    pieces=[]
    start=0
    for n in widths:
        pieces.append(PreparedSeparableFP8(device.linear.weight[start:start+n],
            device.linear.scale[start:start+n],device.linear.bias[start:start+n]))
        start+=n
    return pieces


records = []
saved = {}
grids={'smoke':[(1,2048,(1024,256,256))],
       'main':[(64,2048,(1024,256,256)),(513,2048,(1024,256,256))],
       'tail':[(3,259,(129,64,257))]}
shapes=sum(grids.values(),[]) if args.grid=='all' else grids[args.grid]
for m,k,widths in shapes:
    weights = [(torch.randn(n,k)*64).clamp(-448,448).to(torch.float8_e4m3fn) for n in widths]
    scales = [torch.rand(n)*.02+.01 for n in widths]
    biases = [torch.randn(n)*.1 for n in widths]
    prepared = prepare_fp8_projections(weights,scales,biases)
    x0 = torch.randn(m,k).bfloat16()
    key = f'{m}-{k}-'+ '-'.join(map(str,widths))
    saved[key] = {'checkpoint':weights,'scales':scales,'biases':biases,'input':x0,
                  'prepared_bytes':prepared.linear.weight.view(torch.uint8),
                  'prepared_scales':prepared.linear.scale}
    torch.save(saved,out/'inputs-outputs-oracles.pt')
    device = prepared.to('hpu')
    if args.independent_baseline:
        separate=[p.to('hpu') for p in projection_views(prepared,widths)]
    else:
        separate = None if args.weight_views_inside else projection_views(device,widths)
    x = x0.to('hpu')
    sync()
    for policy in ['per_token_fp8','bf16']:
        activation = x0.double()
        if policy == 'per_token_fp8':
            sa = x0.float().abs().amax(-1,keepdim=True).clamp_min(1e-10)*(1/448)
            qa = (x0.float()/sa).clamp(-448,448).to(torch.float8_e4m3fn).float()
            activation = (qa*.5).to(torch.float8_e4m3fn).double()*(sa.double()*2)
        dot = activation@prepared.linear.weight.double().T*prepared.linear.scale.double()
        reference = (dot+prepared.linear.bias.double()).bfloat16()
        changed_reference = (-dot+prepared.linear.bias.double()).bfloat16()
        saved[key][policy+'_oracle'] = reference
        torch.save(saved,out/'inputs-outputs-oracles.pt')
        baseline = None
        for variant in ['separate','combined_views','combined_contiguous']:
            x.copy_(x0)
            sync()
            if args.warm_input_views:
                if variant=='separate':
                    warm_parts=projection_views(device,widths) if args.weight_views_inside else separate
                    warm=tuple(linear_fp8(x,p,activation=policy,optimized=True) for p in warm_parts)
                else:
                    warm=linear_fp8_projections(x,device,activation=policy,optimized=True)
                    warm_views=split_fp8_projections(warm,device,contiguous=variant=='combined_contiguous')
                sync()
            stream = torch.hpu.Stream()
            graph = torch.hpu.HPUGraph()
            with torch.hpu.graph(graph,stream=stream):
                if variant == 'separate':
                    used_parts=projection_views(device,widths) if args.weight_views_inside else separate
                    ys = tuple(linear_fp8(x,p,activation=policy,optimized=True) for p in used_parts)
                else:
                    y = linear_fp8_projections(x,device,activation=policy,optimized=True)
                    ys = split_fp8_projections(y,device,contiguous=variant=='combined_contiguous')
            sync()
            graph.replay(asynchronous=True)
            sync()
            actual = torch.cat([y.cpu() for y in ys],dim=1)
            error = relative(actual,reference)
            assert bool(torch.isfinite(actual).all()) and error < .001
            if baseline is None: baseline = actual
            saved[key][policy+'_'+variant] = actual
            for _ in range(5): graph.replay(asynchronous=True)
            sync()
            before = counter()
            for _ in range(10): graph.replay(asynchronous=True)
            sync()
            launches = counter()-before
            assert launches == 10
            event,wall = [],[]
            for _ in range(5):
                begin,end = torch.hpu.Event(enable_timing=True),torch.hpu.Event(enable_timing=True)
                clock = time.perf_counter()
                begin.record(stream)
                for _ in range(80): graph.replay(asynchronous=True)
                sync()
                end.record(stream)
                end.synchronize()
                sync()
                event.append(begin.elapsed_time(end)*1000/80)
                wall.append((time.perf_counter()-clock)*1e6/80)
            assert timings_agree(event,wall)
            pointers = [y.data_ptr() for y in ys]
            x.copy_(x0.neg())
            sync()
            graph.replay(asynchronous=True)
            sync()
            changed = torch.cat([y.cpu() for y in ys],dim=1)
            assert relative(changed,changed_reference)<.001 and not torch.equal(changed,actual)
            assert pointers == [y.data_ptr() for y in ys]
            saved[key][policy+'_'+variant+'_changed'] = changed
            row = {'M':m,'K':k,'widths':widths,'activation':policy,'variant':variant,
                   'rounded_contract_relative_l2':error,'checked_elements':actual.numel(),
                   'different_bf16_values_from_separate':int((actual!=baseline).sum()),
                   'output_strides':[list(y.stride()) for y in ys],
                   'event_us':event,'wall_us':wall,'median_event_us':statistics.median(event),
                   'median_wall_us':statistics.median(wall),'launches_per_10_replays':launches,
                   'stable_pointers':True}
            row['warm_input_views']=args.warm_input_views
            row['weight_views_inside']=args.weight_views_inside
            row['independent_baseline']=args.independent_baseline
            records.append(row)
            # Retain partial successes even if a later shape fails.
            torch.save(saved,out/'inputs-outputs-oracles.pt')
            (out/'result.json').write_text(json.dumps({'status':'INCOMPLETE','records':records},indent=2)+'\n')
            print(json.dumps(row),flush=True)
(out/'result.json').write_text(json.dumps({'status':'PASS','records':records,
    'scope':'operator graph only; same prepared bytes; independent loading owners when requested; not model TPS',
    'placement':'postgraph audit still required, especially W8A16 and view consumers'},indent=2)+'\n')
