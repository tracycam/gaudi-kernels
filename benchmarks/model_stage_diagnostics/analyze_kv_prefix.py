"""Independently recheck sealed KV reads/writes and relation to the 70-layer frame."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--case', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(2)
root = a.case.resolve()
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
seal = json.loads((root.parent/(root.name+'-runner')/'remote-sha256.json').read_text())
assert digest(root/'result.json') == seal[root.name+'/result.json']['sha256']
source = json.loads((root/'source/sha256.json').read_text())
for name, expected in source.items():
    assert digest(root/'source'/name) == expected, name
sys.path.insert(0, str(root/'source'))
from kv_prefix_analysis import canonical_kv, compare_kv, prefix_relation, tensor_bits, verify_reference
from teacher_isolation_contract import load_frame, validate_libraries, FRAME_SHA256

result = json.loads((root/'result.json').read_text())
assert result['status'] == 'DIAGNOSTIC' and not result['candidate_accepted']
assert result['serving_prefix_relation'] == 'COMPLETE' and result['kv_analysis_complete']
assert len(result['rows']) == 4
frame = load_frame(root/'source/kv-prefix-frame.json', FRAME_SHA256)
validate_libraries(frame, root/'source')
reference_root = root/'source/kv-prefix-reference'
refs = verify_reference(reference_root, '790ff995baf906b99a01e88112efd8dc326750bef2666dbe485c04df81c54b11')
files = {}


def load(path, expected):
    actual = digest(path)
    assert actual == expected, path
    files[str(path.relative_to(root))] = actual
    return torch.load(path, map_location='cpu', weights_only=False)


relations = []
for name in ('device_old', 'device_routed'):
    plain = next(r for r in result['rows'] if r['name'] == name and not r['detailed'])
    detailed = next(r for r in result['rows'] if r['name'] == name and r['detailed'])
    assert plain['status'] == detailed['status'] == 'RECORDED'
    for rank in range(8):
        boundaries = []
        for row in (plain, detailed):
            entry = next(x for x in row['boundaries'] if x['rank'] == rank)
            boundaries.append(load(root/'kv-prefix'/Path(entry['path']).name, entry['sha256']))
        fields = {k: tensor_bits(boundaries[0][k], boundaries[1][k])
                  for k in ('input_ids', 'positions', 'hidden', 'selected', 'logits')}
        assert all(x['all_bits_equal'] for x in fields.values())
        entry = next(x for x in detailed['prefix_relations'] if x['rank'] == rank)
        current = load(root/'quality-logits'/Path(entry['path']).name, entry['sha256'])
        ref = next(x for x in refs['files'] if x['policy'] == name and x['rank'] == rank)
        previous = load(reference_root/ref['name'], ref['sha256'])
        relation = prefix_relation(previous, current)
        assert relation == entry['relation'] and relation['all_bits_equal']
        relations.append(dict(policy=name, rank=rank, nohook_bits_equal=True,
                              seventy_layer_prefix_bits_equal=True, stages=len(relation['stages'])))


def metrics(x, y):
    assert x.shape == y.shape and x.dtype == y.dtype and x.dtype in (torch.bfloat16, torch.float32)
    delta = (x.float()-y.float()).abs()
    bits = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
    return dict(shape=list(x.shape), dtype=str(x.dtype), different_words=int((x.contiguous().view(bits)!=y.contiguous().view(bits)).sum()),
                max_abs=float(delta.max()), relative_l2=float(delta.norm()/x.float().norm()))


rank_reports = []
checked_words = changed_words = 0
for rank in range(8):
    histories = []
    for name in ('device_old', 'device_routed'):
        row = next(r for r in result['rows'] if r['name']==name and r['detailed'])
        entries = next(r for r in row['finished'] if r['rank']==rank)['kv_files']
        assert len(entries) == 10
        records = []
        for entry in entries:
            remote = Path(entry['path'])
            records.append(load(root/'kv-prefix'/remote.parent.name/remote.name, entry['sha256']))
        histories.append(records)
    raw = compare_kv(*histories)
    assert raw == next(x for x in result['kv_comparisons'] if x['rank']==rank)['report']
    old, new = [canonical_kv(records) for records in histories]
    details = {}
    for kind in ('key', 'value'):
        x, y = old[kind]['values'], new[kind]['values']
        check = metrics(x, y)
        checked_words += x.numel()
        changed_words += check['different_words']
        first = raw[kind]['first_logical_position']
        if first is not None:
            check['first_position'] = first
            check['upstream_l6_at_first'] = metrics(old[kind]['boundaries'][first],new[kind]['boundaries'][first])
        details[kind] = check
    rank_reports.append(dict(rank=rank, calls_per_policy=[len(x) for x in histories],
        valid_write_read_fetch_bits_equal=True, original_comparison=raw, metrics=details))

report = dict(status='ROOT_RECHECKED_KV_DIAGNOSTIC', case=root.name,
    result_sha256=digest(root/'result.json'), frozen_sources_verified=len(source),
    frozen_math_libraries_verified=len(frame['frozen_libraries']), relations=relations,
    checked_active_kv_words_per_policy=checked_words, changed_kv_words=changed_words,
    ranks=rank_reports, tensor_sha256=files, model_quality_qualified=False, performance_qualified=False,
    scope='All active writes/readbacks/fetched KV agree in this fixed frame. Prefill differences already exist in upstream L6 MoE output. This excludes cache copying/mapping corruption here, but does not classify the initial arithmetic change or waive the original 70h quality rejection.')
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: report[k] for k in ('status','checked_active_kv_words_per_policy','changed_kv_words','frozen_math_libraries_verified')}))
