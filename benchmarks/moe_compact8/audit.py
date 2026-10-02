"""Full-output/ABBA and actual compiled graph audit after the bounded gate."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads((a.case/'result.json').read_text());assert r['status']=='PASS_FULL_MOE_PAIRED'
post=a.case/'post_graph.json';paths=[post]if post.is_file()else list(post.rglob('*.json'))
graphs=[g for path in paths for g in json.loads(path.read_text())['graphs']]
# Profiler emits separate files whose graph IDs all restart at zero.
# Match the full recipe by actual output descriptor and physical GUIDs.
def graph_key(graph):
    nodes=graph['nodes'];tensors={t['name']:t for t in graph['tensors']}
    combine=[n for n in nodes if n['guid']=='pf_combine_f32']
    if len(combine)!=1:return None
    shape=tensors[combine[0]['output_tensors'][0]]['max_shape']
    if len(shape)!=2 or shape[0]!=6144:return None
    compact=any(n['guid']=='gk_moe_gp_folded_v1' for n in nodes)
    return (shape[1],'compact' if compact else 'broadcast')

report={'scope':'actual PostGraph placement/physical nodes plus source-requested intervals; no physical HBM claim',
        'graphs':[],'checks':r['checks'],'timings':{}}
for check in r['checks']:
    assert all(v['finite']and v['all_fp32_bits_equal']for v in check['variants'].values())
for record in r['graphs']:
    matches=[g for g in graphs if graph_key(g)==(record['tokens'],record['variant'])]
    assert len(matches)==1,(record,len(matches))
    graph=matches[0];nodes=[n for n in graph['nodes']if not n['is_logical']]
    guids=Counter(n['guid']for n in nodes);tensors={t['name']:t for t in graph['tensors']}
    assert not any(n['engine']=='MME'for n in nodes),'These candidates are per-route GEMV, not grouped MME'
    if record['variant']=='broadcast':
        assert guids['nm_gemv']==2 and guids['nm_prep']==1 and guids['ub_gate_broadcast']==1
        selected=[n for n in nodes if n['guid']=='nm_gemv']
    else:
        assert guids['gk_moe_gp_folded_v1']==1 and guids['downact_direct_down_broadcast']==1 and guids['ub_compact_gate']==1
        assert not guids['nm_prep'] and not guids['ub_gate_broadcast']
        selected=[n for n in nodes if n['guid']in ('gk_moe_gp_folded_v1','downact_direct_down_broadcast')]
    assert guids['pf_combine_f32']==1
    weights=[]
    for n in selected:
        for name in n['input_tensors'][:2]:
            t=tensors[name];assert t['dtype_bit_size']==8 and t['allocation']=='DRAM'
            weights.append(t)
    expected=sorted([603979776,37748736,301989888,18874368])
    assert sorted(math.prod(t['max_shape'])for t in weights)==expected
    report['graphs'].append({**record,'recipe_name':graph['name'],'physical_engines':dict(Counter(n['engine']for n in nodes)),
        'physical_guid_sequence':[n['guid']for n in nodes], 'weight_descriptors':weights,
        'kernel_geometry':[{'guid':n['guid'],'engines':n.get('tpc_working_engines'),'rois':n.get('num_of_ROIs')}for n in selected]})
for name in sorted({x['case']for x in r['timing']}):
    samples=[x for x in r['timing']if x['case']==name]
    for trial in sorted({x['trial']for x in samples}):
        assert [x['variant']for x in samples if x['trial']==trial]==['broadcast','compact','compact','broadcast']
    report['timings'][name]={v:{key:statistics.median(x[key]for x in samples if x['variant']==v)for key in ('event_us','wall_us')}for v in ('broadcast','compact')}
report['status']='AUDITED_NO_DEFAULT_PROMOTION';a.output.write_text(json.dumps(report,indent=2)+'\n')
