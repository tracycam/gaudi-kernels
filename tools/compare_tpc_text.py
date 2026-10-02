"""Check instruction identity after relocating source/build paths; not an HPU test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--candidate',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();records=[]
for name in ['dequant_bf16','activation_native','reduce1','reduce4',
    'dequant_native_f32scale','dequant_native_bf16scale','reduce1bf16','reduce4bf16']:
    left=subprocess.check_output(['readelf','-x','.text',str(args.reference/(name+'.o'))])
    right=subprocess.check_output(['readelf','-x','.text',str(args.candidate/(name+'.o'))])
    records.append({'name':name,'text_identical':left==right,
        'reference_dump_sha256':hashlib.sha256(left).hexdigest(),
        'candidate_dump_sha256':hashlib.sha256(right).hexdigest()})
result={'status':'PASS' if all(r['text_identical'] for r in records) else 'DIFFERENT',
    'scope':'TPC text only; ABI, kernel metadata, glue and runtime behavior require separate verification',
    'records':records}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(result['status']!='PASS')
