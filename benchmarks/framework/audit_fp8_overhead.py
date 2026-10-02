"""Require every W8A16 graph's decoded weight descendants to remain in SRAM."""
import argparse
import json
from pathlib import Path
from audit_w8a16 import audit

p=argparse.ArgumentParser()
p.add_argument('case',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
result=json.loads((a.case/'result.json').read_text())
assert result['status']=='PASS' and result['baseline_optimized_bitwise_equal']
expected=[dict(M=r['M'],N=r['N'],K=r['K'],numerical_pass=r['relative_l2_to_rounded_contract']<.001)
          for r in result['records'] if r['activation']=='bf16']
assert len(expected)==6
expected_path=a.case/'expected-w8a16-fixtures.json'
expected_path.write_text(json.dumps(expected,indent=2)+'\n')
report=audit(a.case/'post_graph.json',expected_path)
assert report['status']=='PASS'
assert all(r['synapse_launches_for_10_replays']==10 for r in result['records'])
report['all_12_variants_one_synapse_launch_per_replay']=True
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'qualified_w8a16_graphs':len(report['records'])}))
