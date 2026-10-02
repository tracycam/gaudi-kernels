"""Read-only CPU audit of sealed 70h tensors; never rewrite acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw(x):
    return x.detach().contiguous().view(torch.uint8).reshape(-1)


def bits(x, y):
    compatible = x.shape == y.shape and x.dtype == y.dtype
    return dict(shape_dtype_match=compatible,
                byte_mismatches=int((raw(x) != raw(y)).sum()) if compatible else None,
                elements=x.numel(), dtype=str(x.dtype), shape=list(x.shape),
                left_sha256=hashlib.sha256(raw(x).numpy().tobytes()).hexdigest(),
                right_sha256=hashlib.sha256(raw(y).numpy().tobytes()).hexdigest())


def compare(reference, candidate):
    aligned = all(torch.equal(reference[k], candidate[k]) for k in ('input_ids', 'positions'))
    x, y = reference['logits'].double(), candidate['logits'].double()
    if x.shape != y.shape:
        raise ValueError('Logit shape mismatch')
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    if not finite:
        raise ValueError('Nonfinite saved logits')
    lp, lq = x.log_softmax(-1), y.log_softmax(-1)
    kl = float((lp.exp() * (lp-lq)).sum(-1).max())
    return dict(same_decode_input=aligned, finite=finite,
                max_kl_ref_to_candidate=kl, passed_existing_kl_gate=aligned and kl <= .01,
                relative_l2_diagnostic=float((x-y).norm()/x.norm().clamp_min(1e-30)),
                top1_match=bool(torch.equal(x.argmax(-1), y.argmax(-1))))


def audit(case):
    source = case/'result.json'
    source_sha = sha(source)
    result = json.loads(source.read_text())
    records = result['teacher_forced']['rows']
    manifest = {'result.json': source_sha}
    captures = {}
    for path in sorted((case/'quality-logits').glob('*.pt')):
        manifest[str(path.relative_to(case))] = sha(path)
        # These are our frozen executor outputs (pickle protocol 4).
        captures[path.name] = torch.load(path, map_location='cpu', weights_only=False)
    rows = {(r['policy'], r['position']): r for r in records}
    def capture(row):
        return captures[Path(row['logits_path']).name]
    reference = result['teacher_forced']['reference_policy']
    candidates = [r for r in records if r.get('quality_role') == 'candidate']
    comparisons = []
    relations = []
    staged = []
    for row in candidates:
        plain = capture(row)
        other = captures[Path(row['audit_logits_path']).name]
        fields = {k: bits(plain[k], other[k]) for k in ('logits', 'input_ids', 'positions')}
        relations.append(dict(position=row['position'], fields=fields,
                              all_bits_equal=all(x['byte_mismatches'] == 0 for x in fields.values())))
        refs = [reference] + [p for p, pos in rows if pos == row['position'] and p.startswith('cpu_')]
        for policy in refs:
            check = compare(capture(rows[policy, row['position']]), plain)
            recorded = row['check'] if policy == reference else next(
                r['check'] for r in result['same_contract_reference']
                if r['reference_policy'] == policy and r['policy'] == row['policy']
                and r['position'] == row['position'])
            if (abs(check['max_kl_ref_to_candidate']-recorded['max_kl_ref_to_candidate']) > 1e-12
                    or check['passed_existing_kl_gate'] != recorded['pass']):
                raise AssertionError('CPU reread did not reproduce the archived gate')
            comparisons.append(dict(reference_policy=policy, candidate_policy=row['policy'],
                                    position=row['position'], archived_gate_reproduced=True, **check))
        entries = [(rank['rank'], entry) for rank in row['same_input_qkv_audit']
                   for entry in rank['same_input_qkv_audit']]
        staged.append(dict(position=row['position'], entries=len(entries),
                           ranks=sorted({rank for rank, _ in entries}),
                           contract_passes=sum(e['fp32_contract']['passed'] for _, e in entries),
                           plain_staged_bit_passes=sum(e['plain_vs_staged']['all_bits_equal'] for _, e in entries),
                           unique_rank_layer_pairs=len({(rank, e['layer']) for rank, e in entries}),
                           scope='Saved scalar/hash contract records, not retained stage tensors or cross-policy inputs'))
    adaptation = []
    for row in result['native_half_adaptation']:
        adaptation.append(dict(position=row['position'], **compare(
            capture(rows[row['original_policy'], row['position']]),
            capture(rows[row['native_policy'], row['position']]))))
    for name in ('git-revision.json', 'native_service_test.py', 'production_quality.py',
                 'sitecustomize.py', 'layer_diagnostics.py',
                 'production-runtime/kernels/production_integration.py'):
        path = case/'source'/name
        if path.exists():
            manifest[str(path.relative_to(case))] = sha(path)
    if sha(source) != source_sha:
        raise RuntimeError('Result changed during read-only audit')
    stage_paths = [p for r in records for p in r['stage_paths']]
    stage_files = sorted(str(p.relative_to(case)) for p in (case/'quality-logits').glob('*stages-rank*.pt'))
    return dict(case_name=case.name, result_status=result['status'],
                candidate_accepted=result['candidate_accepted'], torch_version=torch.__version__,
                teacher_rows=len(records), quality_tensor_files=len(captures),
                nonempty_stage_paths=stage_paths, actual_stage_files=stage_files,
                first_differing_layer='UNAVAILABLE: no saved per-layer tensors or KV snapshots',
                comparisons=comparisons, plain_vs_staged=relations, staged_qkv=staged,
                native_half_vs_ocp=adaptation, sha256=manifest,
                scope='CPU reread of existing evidence only; preserves 70h REJECTED and KL<=0.01; no device or acceptance mutation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.case)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: report[k] for k in ('result_status', 'candidate_accepted', 'teacher_rows',
          'quality_tensor_files', 'nonempty_stage_paths', 'first_differing_layer')}))


if __name__ == '__main__':
    main()
