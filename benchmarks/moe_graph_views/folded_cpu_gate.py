"""Check provenance rejection and byte-identical generation of SIM-qualified text."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import shutil

p=argparse.ArgumentParser();p.add_argument('--gp-dir',type=Path,required=True);p.add_argument('--down-dir',type=Path,required=True)
p.add_argument('--offline-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
from moe_activation_fold_core import PINS,embedded_elf,fold,sha,verify_deployed
result={'status':'STARTED','device_used':False,'targets':[],'rejections':[]}
for target,directory in [('gp',a.gp_dir),('down',a.down_dir)]:
 data,provenance=verify_deployed(directory,target);text,schedule=fold(data['assembly'].decode())
 prior=a.offline_archive/(target+'-fold-a')/'candidate.s';assert text.encode()==prior.read_bytes()
 result['targets'].append({'target':target,'deployed':provenance,'exact_previous_SIM_assembly':True,'candidate_assembly_sha256':sha(text.encode()),'schedule':schedule})
 with tempfile.TemporaryDirectory() as temp:
  d=Path(temp)
  for kind in ['library','elf','assembly']:shutil.copyfile(directory/PINS[target][kind],d/PINS[target][kind])
  for kind in ['library','elf','assembly']:
   path=d/PINS[target][kind];original=path.read_bytes();path.write_bytes(original[:-1]+bytes([original[-1]^1]))
   try:verify_deployed(d,target)
   except ValueError:result['rejections'].append(target+'_'+kind+'_tamper')
   else:raise AssertionError('modified deployed input accepted')
   path.write_bytes(original)
 try:embedded_elf(data['library'],'missing')
 except ValueError:result['rejections'].append(target+'_missing_embedded_symbol')
 else:raise AssertionError('missing ELF accepted')
 try:fold(data['assembly'].decode().replace('ld_tnsr V39, 0x2, I6','ld_tnsr V40, 0x2, I6'))
 except (ValueError,StopIteration,AssertionError):result['rejections'].append(target+'_changed_schedule')
 else:raise AssertionError('modified schedule accepted')
result['status']='CPU_PASS_NOT_DEVICE_QUALIFIED';a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'rejections':len(result['rejections'])}))
