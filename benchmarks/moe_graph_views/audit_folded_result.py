"""Offline verification of the precise ELF and physical graph used in ABBA."""
import argparse
import collections
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from moe_activation_fold_core import embedded_elf,sha,verify_deployed
from moe_folded_plan import ELFS
d=a.archive;r=d/'results/folded-paired-a';result=json.loads((r/'result.json').read_text());launch=json.loads((r/'exit.json').read_text())
assert result['status']=='PASS_BITWISE_AND_REPLAY' and launch['runner_exit_code']==0
assert launch['module']==5 and launch['bus_id']=='0000:cd:00.0' and launch['source_identity_verified']
build=json.loads((d/'builds/folded-a/build.json').read_text());plan=json.loads((r/'folded-plan.json').read_text())
assert build['source_commit']==launch['git_commit']==plan['build_source_commit']
report={'status':'AUDITED','source_commit':launch['git_commit'],'compiled_graphs':[],
        'elfs':{},'paired_trials':{},'checked_values':0,'device_profile_collected':False,
        'limits':['HPU event span includes submission gaps','PostGraph placement is not a physical traffic counter',
                  'Static packets are not measured cycles','Stable logical handles are not physical-address invariance'],
        'graph_comparison':'Preserve fused content hash and dtype; normalize only compiler-assigned fused-kernel instance serial'}
for kind in ('gp','down'):
    deployed,_=verify_deployed(d/('builds/qualified-'+kind),kind)
    candidate=(d/('builds/folded-a/folded_'+kind+'.o')).read_bytes()
    library=(d/'builds/folded-a/libgaudi_moe_activation_folded_tpc.so').read_bytes()
    assert sha(candidate)==ELFS[kind] and embedded_elf(library,'folded_'+kind)==candidate
    assert sha(library)==launch['kernel_libraries_sha256']['builds/folded-a/libgaudi_moe_activation_folded_tpc.so']
    with tempfile.TemporaryDirectory() as td:
        dest=Path(td)/'actual.text'
        subprocess.run(['objcopy','--dump-section','.text='+str(dest),str(d/('builds/folded-a/folded_'+kind+'.o'))],check=True)
        assert dest.read_bytes()==(d/('builds/folded-a/folded_'+kind+'.text')).read_bytes()
        text_sha=sha(dest.read_bytes())
    assert text_sha==build['targets'][kind]['candidate_text_sha256']
    report['elfs'][kind]={'embedded_sha256':sha(candidate),'actual_text_sha256':text_sha,
                           'schedule':build['targets'][kind]['schedule']}
graphs={g['id']:g for g in json.loads((r/'post_graph.json').read_text())['graphs']}
alias={'gk_moe_gp_folded_v1':'diag_direct_gp_broadcast','gk_moe_down_folded_v1':'downact_direct_down_broadcast'}
for record in result['records']:
    reference=None
    for item in record['graphs']:
        assert len(item['postgraph_ids'])==1;g=graphs[item['postgraph_ids'][0]]
        nodes=[n for n in g['nodes'] if not n['is_logical']]
        sequence=[(n['engine'],re.sub(r'^(fused_kernel_0x[0-9A-F]+)_[0-9A-F]+_(.+)$',r'\1_INSTANCE_\2',alias.get(n['guid'],n['guid']))) for n in nodes]
        if reference is None:reference=sequence
        else:assert reference==sequence,'unexpected physical node change'
        expected=[('gk_moe_gp_folded_v1' if item['variant'] in ('folded_gp_only','folded_both') else 'diag_direct_gp_broadcast'),
                  ('gk_moe_down_folded_v1' if item['variant'] in ('folded_down_only','folded_both') else 'downact_direct_down_broadcast')]
        selected=[n for n in nodes if n['guid'] in alias or n['guid'] in alias.values()]
        assert [n['guid'] for n in selected]==expected
        details=[];tensors={t['name']:t for t in g['tensors']}
        for n in selected:
            assert n['tpc_working_engines']==[24]
            details.append({'guid':n['guid'],'tpc_working_engines':n['tpc_working_engines'],
                            'inputs':[tensors[v] for v in n['input_tensors']],
                            'outputs':[tensors[v] for v in n['output_tensors']]})
        report['compiled_graphs'].append({'case':record['case'],'variant':item['variant'],'id':g['id'],
            'physical_engines':dict(collections.Counter(n['engine'] for n in nodes)),
            'physical_guid_sequence':[(n['engine'],n['guid']) for n in nodes],'selected_nodes':details})
    report['paired_trials'][record['case']]={}
    for candidate in record['pair_median_us']:
        deltas=[]
        for trial in range(3):
            samples=[v for v in record['timing'] if v['paired_candidate']==candidate and v['trial']==trial]
            assert [v['variant'] for v in samples]==['grouped_control',candidate,candidate,'grouped_control']
            deltas.append({key:statistics.mean(v[key] for v in samples if v['variant']=='grouped_control')-
                statistics.mean(v[key] for v in samples if v['variant']==candidate) for key in ('event_us','wall_us')})
        report['paired_trials'][record['case']][candidate]={'median_us':record['pair_median_us'][candidate],
                                                            'per_trial_control_minus_candidate_us':deltas}
    for check in record['checks']:
        assert all(check['bitwise_variants_vs_baseline']) and check['replays_per_variant']==10
        report['checked_values']+=check['checked_values']
assert report['checked_values']==1081344
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'graphs':len(report['compiled_graphs']),'checked_values':report['checked_values']}))
