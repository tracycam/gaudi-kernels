#!/usr/bin/env python3
"""Read-only complete-record audit of the two-layer live v1 preflight."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case',type=Path,required=True)
parser.add_argument('--executor',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.executor/'executor'))
from fp32_quality_contract import classify, staged_gate
result=json.loads((args.case/'result.json').read_text())
assert result['status']=='DIAGNOSTIC' and result['candidate_accepted'] is False
rows=[]
for row in result['teacher_forced']['rows']:
 if row.get('quality_role')!='candidate':continue
 gate=staged_gate(row,2);assert gate['passed'] and gate['checked_layers']==16
 records=[]
 for worker in row['same_input_qkv_audit']:
  for record in worker['same_input_qkv_audit']:
   report=record['fp32_contract'];stages=report['stages'];loss=report['representation_loss']
   assert loss['category']=='native_half_adaptation_not_fp32_accumulation'
   assert loss['weight_changed_values']==record['rounded_half_values']
   assert all(stages[key]['match'] for key in ('q_native','activation_scales','prepared_weight','prepared_scales','bias','epilogue'))
   assert stages['partial']['passed'] and stages['partial']['incompatible_cells']==0
   assert record['plain_vs_staged']['all_bits_equal'] and report['passed'] and report['numerical_passed'] and report['implementation_qualified']
   assert record['reduction_policy']==report['reduction']=='neumaier_isa_fp32'
   assert record['source_layout']['local_weight_shape']==[3392,6144] and record['source_layout']['local_scale_shape']==[27,48]
   records.append(dict(rank=worker['rank'],layer=record['layer'],input_sha256=record['input_sha256'],
       partial=stages['partial'],adaptation=loss,plain_vs_staged=record['plain_vs_staged'],diagnostic_wall_s=record['diagnostic_wall_s']))
 rows.append(dict(policy=row['policy'],position=row['position'],gate=gate,
                  plain_vs_audit_logits=row['plain_vs_audit_logits'],records=records))
rechecked=classify(result,result['quality_plan'],(0,),2)
assert all(rechecked[key] for key in ('candidate_teacher_pass','staged_qkv_pass','matched_fp32_reference_pass'))
assert rechecked['normal_eos_and_tasks_pass'] is False and rechecked['passed'] is False
report=dict(case_name=args.case.name,result_sha256=hashlib.sha256((args.case/'result.json').read_bytes()).hexdigest(),
 executor_source=json.loads((args.case/'source/git-revision.json').read_text()),
 classifier_sha256=hashlib.sha256((args.executor/'executor/fp32_quality_contract.py').read_bytes()).hexdigest(),
 rechecked=rechecked,rows=rows,scope='all16live records; h remains two-layer DIAGNOSTIC, no original-OCP or normal-EOS/full-model qualification')
assert not args.out.exists();args.out.write_text(json.dumps(report,indent=2)+'\n')
print('PASS: all16layer records and current reducer-policy binding; expected incomplete full-model gate')
