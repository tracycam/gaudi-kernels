"""Offline attribution bounds for trusted, executor-owned stage captures.

No device access and no altered acceptance threshold. Same-input observations
are distinct from accumulated trajectory errors; missing stages fail closed.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

LAYER = re.compile(r'^(.*\.layers\.(\d+))\.(.+)$')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(x, y):
    if x.shape != y.shape or x.dtype != y.dtype:
        raise ValueError('Shape/dtype mismatch')
    dx, dy = x.double(), y.double()
    delta = dy - dx
    finite = bool(torch.isfinite(dx).all() and torch.isfinite(dy).all())
    row = dict(shape=list(x.shape), dtype=str(x.dtype), elements=x.numel(),
               finite=finite, different_elements=int((x != y).sum()),
               bitwise_different_elements=int((x.contiguous().view(torch.uint8).reshape(x.numel(),x.element_size()) !=
                   y.contiguous().view(torch.uint8).reshape(y.numel(),y.element_size())).any(1).sum()),
               max_abs=float(delta.abs().max()),
               reference_l2=float(dx.norm()), delta_l2=float(delta.norm()),
               relative_l2=float(delta.norm()/dx.norm().clamp_min(1e-30)))
    if x.dtype == torch.bfloat16 and finite:
        ix, iy = [z.contiguous().view(torch.int16).to(torch.int32) & 65535 for z in (x, y)]
        keys = [torch.where(z & 32768 != 0, 32768-(z & 32767), 32768+z) for z in (ix, iy)]
        ulp = (keys[1]-keys[0]).abs()
        row.update(bitwise_different_elements=int((ix != iy).sum()),
                   max_bf16_ulp_distance=int(ulp.max()),
                   different_by_more_than_one_bf16_ulp=int((ulp > 1).sum()))
    return row


def load(path):
    # Only files produced by our own pinned executor; never an untrusted checkpoint.
    before = digest(path)
    data = torch.load(path, map_location='cpu', weights_only=False)
    if before != digest(path):
        raise ValueError('Capture changed during analysis: '+str(path))
    for key in ('stages', 'input_ids', 'positions'):
        if key not in data:
            raise ValueError('Missing capture key '+key)
    return data, dict(path=str(path), bytes=path.stat().st_size, sha256=before)


def compare(reference, candidate, expected_layers):
    for key in ('input_ids', 'positions'):
        if not torch.equal(reference[key], candidate[key]):
            raise ValueError('Teacher input mismatch: '+key)
    xs, ys = reference['stages'], candidate['stages']
    if list(xs) != list(ys):
        raise ValueError('Stage names or execution order differ')
    indices = {int(LAYER.match(name).group(2)) for name in xs if LAYER.match(name)}
    if indices != set(range(expected_layers)):
        raise ValueError('Incomplete layer coverage: '+str(sorted(indices)))
    rows, routes = [], []
    for name, x in xs.items():
        y = ys[name]
        match = LAYER.match(name)
        if not match:
            raise ValueError('Unknown layer stage '+name)
        prefix, layer, suffix = match.groups()
        row = dict(name=name, layer=int(layer), stage=suffix, **metric(x, y))
        if suffix == 'self_attn.qkv_proj':
            before = prefix+'.input_layernorm'
            row['observed_qkv_input_equal'] = torch.equal(xs[before], ys[before])
            row['interpretation'] = ('local output difference on equal observed normalized input'
                if row['observed_qkv_input_equal'] and row['different_elements'] else
                'trajectory comparison; local kernel error not isolated')
        if name.endswith('.route_ids'):
            # IDs are categorical. Their arithmetic L2 above is not an error severity measure.
            xr, yr = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
            wr, wc = xs[name.replace('.route_ids', '.route_weights')], ys[name.replace('.route_ids', '.route_weights')]
            wr, wc = wr.reshape_as(xr), wc.reshape_as(yr)
            route = dict(name=name, layer=int(layer), ordered_ids_match=torch.equal(x, y),
                expert_sets_match=torch.equal(x.sort(-1).values, y.sort(-1).values),
                reference_ids=x.tolist(), candidate_ids=y.tolist(), token_rows=[])
            for token, (a, b) in enumerate(zip(xr.tolist(), yr.tolist())):
                common = sorted(set(a) & set(b))
                route['token_rows'].append(dict(token_row=token, intersection=len(common),
                    removed=sorted(set(a)-set(b)), added=sorted(set(b)-set(a)),
                    aligned_common_weight_max_abs=max((abs(float(wr[token,a.index(e)])-float(wc[token,b.index(e)])) for e in common), default=None)))
            gate = prefix+'.mlp.gate'
            route['router_logits'] = metric(xs[gate], ys[gate])
            route['biased_group_topk_margin_available'] = False
            routes.append(route)
            row.update(ordered_ids_match=route['ordered_ids_match'], expert_sets_match=route['expert_sets_match'])
        rows.append(row)
    return dict(rows=rows, routes=routes,
        first_observed_difference=next((r for r in rows if r['different_elements']), None),
        first_route_order_difference=next((r for r in routes if not r['ordered_ids_match']), None),
        first_route_set_difference=next((r for r in routes if not r['expert_sets_match']), None),
        changed_route_sets=sum(not r['expert_sets_match'] for r in routes),
        changed_route_orders=sum(not r['ordered_ids_match'] for r in routes),
        route_layers=len(routes))


def replicated_stages(captures):
    names = list(captures[0]['stages'])
    selected = [n for n in names if LAYER.match(n).group(3) in
        ('input','input_layernorm','self_attn.o_proj','self_attn','post_attention_layernorm',
         'mlp.gate','mlp.experts','mlp','output','residual') or n.endswith(('.route_ids','.route_weights'))]
    for cap in captures:
        if list(cap['stages']) != names:
            raise ValueError('Stage schemas differ across ranks')
    differences = []
    for name in selected:
        x = captures[0]['stages'][name]
        for rank, cap in enumerate(captures[1:], 1):
            y = cap['stages'][name]
            if not torch.equal(x, y):
                differences.append(dict(name=name, rank=rank, **metric(x,y)))
    return dict(assumption='listed outputs are replicated TP values; QKV/RoPE/attention shards excluded',
                stages_checked=len(selected), rank0_vs_other_comparisons=len(selected)*7,
                differences=differences)


def run(args):
    torch.set_num_threads(1)
    pairs, sources = [], []
    all_reference, all_candidate = [], []
    for rank in range(8):
        captures=[]
        for tag in (args.reference,args.candidate):
            data, identity = load(args.case/'quality-logits'/f'{tag}-teacher-0-stages-rank{rank}.pt')
            captures.append(data); sources.append(identity)
        pair=compare(*captures,args.expected_layers); pair['rank']=rank; pairs.append(pair)
        all_reference.append(captures[0]); all_candidate.append(captures[1])
    layer_rows=[]
    for index in range(args.expected_layers):
        for suffix in ('input','input_layernorm','self_attn.qkv_proj','self_attn.attn',
                       'self_attn.o_proj','post_attention_layernorm','mlp.gate','mlp','output','residual'):
            records=[r for p in pairs for r in p['rows'] if r['layer']==index and r['stage']==suffix]
            if records:
                layer_rows.append(dict(layer=index,stage=suffix,ranks=len(records),
                    max_rank_relative_l2=max(r['relative_l2'] for r in records),
                    max_rank_abs=max(r['max_abs'] for r in records),
                    total_different_elements=sum(r['different_elements'] for r in records),
                    equal_input_ranks=sum(r.get('observed_qkv_input_equal',False) for r in records)))
    report=dict(reference=args.reference,candidate=args.candidate,expected_layers=args.expected_layers,
        all_stage_bits_equal=not any(r['bitwise_different_elements'] for p in pairs for r in p['rows']),
        input_ids=all_reference[0]['input_ids'].tolist(),positions=all_reference[0]['positions'].tolist(),
        sources=sources,ranks=pairs,per_layer_max_rank=layer_rows,
        reference_rank_consistency=replicated_stages(all_reference),
        candidate_rank_consistency=replicated_stages(all_candidate),
        acceptance_thresholds_changed=False,
        scope='Offline teacher-0 observations only; not a performance measurement or proof that discrepancies are unavoidable',
        limitations=['CPU FP64 MAC then BF16 is an accuracy reference, not device FP32 reduction emulation.',
          'Equal teacher token/position alone does not prove identical KV state or intermediate inputs.',
          'Only QKV has a directly observable normalized input in this schema; other local errors are not isolated.',
          'Correction bias, group selection scores, and route-selection margins are not captured.',
          'Final layer.output is the MLP branch and layer.residual is the residual branch; neither alone is the final normalized hidden state.',
          'No raw activation quantizer bytes/scales, block partial sums, or per-expert GP/down/combine tensors are captured.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(rank=p['rank'],first_difference=p['first_observed_difference']['name'] if p['first_observed_difference'] else None,
        first_route_set_difference=p['first_route_set_difference']['name'] if p['first_route_set_difference'] else None,
        changed_route_sets=p['changed_route_sets']) for p in pairs],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--reference',required=True)
    p.add_argument('--candidate',required=True);p.add_argument('--expected-layers',type=int,default=70);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
