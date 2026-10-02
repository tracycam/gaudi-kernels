"""Reduce sealed device records without reclassifying failed candidates as passes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import torch

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
report=dict(device_cases=[],placement=[],model_quality_qualified=False,physical_HBM_measured=False)
for path in sorted(a.root.glob('*/results/*/result.json')):
    data=json.loads(path.read_text());exit_data=json.loads((path.parent/'exit.json').read_text())
    record=dict(path=str(path.relative_to(a.root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status=data.get('status', 'PASS' if data.get('all_pass') else 'FAIL'),
        runner_exit=exit_data['runner_exit_code'],elapsed_s=exit_data['elapsed_s'],
        preflight_returncode=exit_data['preflight']['returncode'],postflight_returncode=exit_data['postflight']['returncode'],
        source_commit=exit_data['git_commit'],kernel_libraries=exit_data['kernel_libraries_sha256'])
    for key in ('checked','bit_mismatches','candidate_nan','reference_unchanged'):
        if key in data:record[key]=data[key]
    direct=path.parent/'direct-gp-fp32.pt'
    if direct.exists():
        # These are this experiment's locally hash-verified, self-produced artifacts.
        records=torch.load(direct,weights_only=False);record['direct_fp32']=[]
        for row in records:
            old=row['old'].contiguous().view(torch.int32).flatten();new=row['candidate'].contiguous().view(torch.int32).flatten()
            bad=(old!=new).nonzero().flatten()
            item=dict(rows=row['x'].shape[0],pairs=old.numel(),mismatches=bad.numel(),candidate_nan=int(torch.isnan(row['candidate']).sum()),old_finite=bool(torch.isfinite(row['old']).all()))
            if bad.numel():
                idx=int(bad[0]);item['first']=dict(index=idx,old_bits=hex(int(old[idx])&0xffffffff),candidate_bits=hex(int(new[idx])&0xffffffff),old_value=float(row['old'].flatten()[idx]))
            record['direct_fp32'].append(item)
    report['device_cases'].append(record)
report['experiment_all_pass']=all(r['runner_exit']==0 for r in report['device_cases'])
final=a.root/'module7-a611991/results/gp-scale-tail-loadsafe-gate-a'
data=json.loads((final/'result.json').read_text())
assert data['status']=='PASS_BITWISE_AND_REPLAY' and all(x['bitwise_equal'] for x in data['direct_gp_fp32_checks'])
assert all(all(c['bitwise_variants_vs_baseline']) for r in data['records'] for c in r['checks'])
assert json.loads((final/'exit.json').read_text())['runner_exit_code']==0
report['final_candidate_pass']=True
graphs={x['id']:x for x in json.loads((final/'post_graph.json').read_text())['graphs']}
report['direct_fp32_returned_values_both_arms']=sum(x['checked_values'] for x in data['direct_gp_fp32_checks'])
report['whole_chain_returned_fp32_values_both_arms']=sum(x['checked_values'] for r in data['records'] for x in r['checks'])
report['unique_pair_comparisons']=(report['direct_fp32_returned_values_both_arms']+report['whole_chain_returned_fp32_values_both_arms'])//2
report['whole_chain_timing']=[dict(rows=r['rows'],median_us=r['median_us'],samples_per_arm=6,replays_per_sample=16) for r in data['records']]
report['M1_FP64_reference_max_relative_l2']=max(c['baseline_vs_fp64_mac_reference']['relative_l2'] for r in data['records'] for c in r['checks'] if 'baseline_vs_fp64_mac_reference' in c)
for record in data['records']:
    pair=[graphs[g['postgraph_ids'][0]] for g in record['graphs']]
    def signature(g):
        def normalized(guid):
            guid=guid.replace('gk_moe_gp_scale_tail_v1','gk_moe_gp_folded_v1')
            return re.sub(r'^(fused_kernel_0x[0-9A-F]+)_[0-9A-F]+_',r'\1_ID_',guid)
        return [(normalized(n['guid']),n['engine'],n.get('tpc_working_engines')) for n in g['nodes'] if not n['is_logical']]
    assert signature(pair[0])==signature(pair[1]);assert pair[0]['workspace_size']==pair[1]['workspace_size']
    entries=[]
    for g in pair:
        ts={t['name']:t for t in g['tensors']}
        gp=next(n for n in g['nodes'] if n['guid'].startswith('gk_moe_gp'))
        tensors=lambda names:[{k:ts[name][k] for k in ('dtype','max_shape','allocation','persistent')} for name in names]
        entries.append(dict(guid=gp['guid'],engines=gp['tpc_working_engines'],inputs=tensors(gp['input_tensors']),outputs=tensors(gp['output_tensors'])))
    assert entries[0]['inputs']==entries[1]['inputs'] and entries[0]['outputs']==entries[1]['outputs']
    report['placement'].append(dict(rows=record['rows'],physical_nodes=len(signature(pair[0])),workspace_bytes=pair[0]['workspace_size'],same_normalized_physical_node_sequence=True,gp=entries))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('experiment_all_pass','final_candidate_pass','direct_fp32_returned_values_both_arms','whole_chain_returned_fp32_values_both_arms','unique_pair_comparisons','M1_FP64_reference_max_relative_l2')}))
