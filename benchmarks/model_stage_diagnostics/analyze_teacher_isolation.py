"""CPU recheck of sealed fixed-frame teacher isolation, never model promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from audit_70h_quality_scope import compare, bits

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--case',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(2)
result=json.loads((a.case/'result.json').read_text())
assert result['status']=='DIAGNOSTIC' and not result['candidate_accepted']
assert len(result['rows'])==6 and result['plain_staged_relation_status']=='COMPLETE'
manifest=json.loads((a.case.parent/(a.case.name+'-runner')/'remote-sha256.json').read_text())
digests={}
def load(path,expected):
    actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==expected
    digests[str(path.relative_to(a.case))]=actual
    return torch.load(path,map_location='cpu',weights_only=False)
captures={};relations=[]
for row in result['rows']:
    payload=load(a.case/'quality-logits'/Path(row['logits_path']).name,row['logits_sha256'])
    assert payload['input_ids'].tolist()==[[279]] or payload['input_ids'].tolist()==[279]
    assert payload['positions'].numel()==1 and payload['positions'].item()==4112
    if row['staged']:
        fields={k:bits(captures[row['name']][k],payload[k])for k in ('input_ids','positions','logits')}
        assert all(v['byte_mismatches']==0 for v in fields.values())
        relations.append(dict(name=row['name'],all_bits_equal=True,fields=fields))
    else:captures[row['name']]=payload
comparisons=[]
for row in result['comparisons']:
    check=compare(captures[row['reference']],captures[row['candidate']])
    assert abs(check['max_kl_ref_to_candidate']-row['check']['max_kl_ref_to_candidate'])<1e-12
    assert check['passed_existing_kl_gate']==row['check']['pass']
    comparisons.append(dict(reference=row['reference'],candidate=row['candidate'],**check))
sys.path.insert(0,str(a.case/'source'))
from analyze_layer_diagnostics import compare as compare_stages
staged=[r for r in result['rows']if r['staged']]
layers=[]
for rank in range(8):
    files=[next(f for f in r['stage_files']if f['rank']==rank)for r in staged]
    payloads=[load(a.case/'quality-logits'/Path(f['path']).name,f['sha256'])for f in files]
    rows=compare_stages(*payloads)
    first=next((i for i,r in enumerate(rows)if r['different_elements']),None)
    assert first is not None
    archived=next(r for r in result['layer_comparison']['ranks']if r['rank']==rank)
    assert rows[first]['name']==archived['first_difference']
    layers.append(dict(rank=rank,first_difference=rows[first],
        preceding_stages_equal=rows[max(0,first-5):first],
        first_route_set_difference=next((r for r in rows if r.get('expert_sets_match')is False),None),
        differing_stages=sum(r['different_elements']>0 for r in rows),stage_count=len(rows),
        output_error_by_layer=[r for r in rows if r['name'].endswith('.output')]))
report=dict(status='ROOT_RECHECKED_DIAGNOSTIC',case=a.case.name,
    result_sha256=hashlib.sha256((a.case/'result.json').read_bytes()).hexdigest(),
    plain_staged_relations=relations,comparisons=comparisons,rank_layers=layers,tensor_sha256=digests,
    model_quality_qualified=False,performance_qualified=False,
    scope='70-layer, teacher position16 only. Both paths rebuild their own KV. Same-input decode QKV references do not certify routed prefill or GPU equivalence. Existing70h rejection and KL gate remain unchanged.')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(comparisons=comparisons,first_differences=[dict(rank=r['rank'],first=r['first_difference'],first_route_set=r['first_route_set_difference'])for r in layers])))
