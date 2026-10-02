"""Attribute full native engine envelopes to the gate's exact replay sequence."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qkv_postprocess'))
from analyze_chain_profile import analyze

p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
p.add_argument('--graph-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
gate=json.loads((a.case/'result.json').read_text());assert gate['profile_only'] and gate['status']=='PASS_FULL_MOE_PAIRED'
audit=json.loads(a.graph_audit.read_text());mapping={g['recipe_name']:(g['tokens'],g['variant'])for g in audit['graphs']}
trace=list((a.case/'trace').glob('*.json'));assert len(trace)==1
r=analyze(trace[0],a.case/'post_graph.json',list(mapping))
ordered=sorted(r['invocations'],key=lambda x:x['start_us'])
expected=[sample for sample in gate['timing'] for _ in range(sample['replays'])]
# Source has no replay between the final correctness gate and the timing loop.
# Prefix counts include one capture execution, one initial replay, and four
# correctness states with four replays each. This is asserted, not guessed by duration.
count=len(expected);prefix=ordered[:-count];timed=ordered[-count:]
assert Counter(x['recipe']for x in prefix)==Counter({name:18 for name in mapping})
assert len(timed)==count==72
for row,sample in zip(timed,expected):
    tokens,variant=mapping[row['recipe']]
    assert variant==sample['variant'] and sample['case'].startswith(f't{tokens}-')
    row['case']=sample['case'];row['trial']=sample['trial'];row['variant']=variant
summary={}
for case in sorted({x['case']for x in timed}):
    summary[case]={}
    for variant in ('broadcast','compact'):
        rows=[x for x in timed if x['case']==case and x['variant']==variant]
        assert len(rows)==6
        ops=defaultdict(list);cores=defaultdict(set)
        for row in rows:
            seen=Counter()
            for n in row['nodes']:
                op=n['op'];seen[op]+=1;key=op+f'#{seen[op]}'
                ops[key].append(n['envelope_us']);cores[key].add(n['engine_count'])
        summary[case][variant]={'native_envelope_us':statistics.median(x['envelope_us']for x in rows),
            'invocations':len(rows),'physical_nodes':rows[0]['physical_nodes'],
            'node_envelope_median_us':{k:statistics.median(v)for k,v in ops.items()},
            'observed_engine_counts':{k:sorted(v)for k,v in cores.items()}}
r['timed_invocations']=timed;r['summary']=summary
r['sequence_assignment']='Actual recipe identity plus exact source replay order; 18 qualification invocations per recipe then 72 timed invocations, all counts and ABBA identities asserted.'
a.output.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(summary,indent=2))
