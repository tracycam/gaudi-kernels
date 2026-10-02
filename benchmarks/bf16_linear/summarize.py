"""Produce a small checked-in index while preserving all raw probe assets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--builds',type=Path,required=True);a=p.parse_args()
summary={'scope':'synthetic BF16 operator coverage; no model acceptance or physical HBM traffic measurement','cases':[],'instruction_audit':[]}
for path in sorted(a.evidence.glob('bf16-*/result.json')):
 data=json.loads(path.read_text());records=data['records'];summary['cases'].append({'case':path.parent.name,'binary_sha256':data['binary_sha256'],'configurations':len(records),'failed':[r['label'] for r in records if r['returncode'] or not r.get('correctness',{}).get('screen_pass')],'total_output_elements_checked':sum(r.get('correctness',{}).get('checked',0) for r in records),'records':records})
for path in sorted(a.builds.glob('*/*.s')):
 code=path.read_text().split('.type\ttpc_compiler')[0]
 summary['instruction_audit'].append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bf16_acc_fp32_instructions':len(re.findall(r'mac\.bf16 acc_fp32',code)),'f32_mac_instructions':len(re.findall(r'mac\.f32',code)),'bf16_load_conversions':len(re.findall(r'convert\.bf16',code)),'final_fp32_to_bf16_conversion':len(re.findall(r'convert\.f32.*target_type=bf16',code)),'vector_local_spill_instructions':re.findall(r'^.*(?:ld_l_v|st_l_v).*$',code,re.M),'scope':'static generated instruction counts; loop pragma is not proof of unrolling'})
(a.evidence/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
files={}
for base in [a.evidence,a.builds]:
 for path in sorted(base.rglob('*')):
  if path.is_file() and path.name not in {'SHA256SUMS.json','.gitignore'}:files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
(a.evidence/'SHA256SUMS.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps({'cases':[(v['case'],v['configurations'],len(v['failed'])) for v in summary['cases']],'files_hashed':len(files)}))
