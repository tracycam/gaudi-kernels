"""Bounded native-ISA gates, explicitly TPC_RUNNER=0 and no device API."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from reference import reference

p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);build=a.build.resolve();records=[]
cases=[('tails',[[3,1],[1,2],[3,1]],4,2,5),('negative',[[-1,1],[2,3]],4,2,4),('duplicate',[[1,1],[2,3]],4,2,4),('overflow',[[0,1],[2,3]],4,2,1),('t513',[[i%4,(i+1)%4]for i in range(513)],4,16,67),('e384',[[383,2,381,4,379,6,377,8]],384,1,8)]
for length in (63,64,65,127,128,129):
 cases.append(('flat'+str(length),[[i%4]for i in range(length)],4,min(length,16),min(4*((length+15)//16),4+(length-4)//16)))
for state in ('uniform','hot','skew'):
 ids=[[(t*8+r)%384 for r in range(8)]for t in range(513)]
 if state!='uniform':
  for t in range(513 if state=='hot'else 384):ids[t]=list(range(376,384))
 cases.append(('T513E384-'+state,ids,384,16,616))
for name,ids,e,c,b in cases:
 out=a.output/name;out.mkdir();input_path=out/'ids.bin';np.asarray(ids,dtype=np.int32).tofile(input_path)
 cmd=[str(build/'simulator'),str(build/'libgaudi_route_metadata_v2_tpc.so'),str(input_path.resolve()),str(out.resolve()),str(len(ids)),str(len(ids[0])),str(e),str(c),str(b)]
 with (out/'sim.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,TPC_RUNNER='0'),timeout=45,check=True)
 want=reference(ids,e,c,b);mismatches={}
 for field,values in want.items():
  values=np.asarray(values,dtype=np.int32).reshape(-1)
  got=np.fromfile(out/(field+'.bin'),dtype=np.int32);np.asarray(values,dtype=np.int32).tofile(out/(field+'-cpu.bin'));assert len(got)==len(values);mismatches[field]=int(np.count_nonzero(got!=values))
 records.append(dict(case=name,status=want['status'][0],mismatches=mismatches,command=cmd));(a.output/'partial-result.json').write_text(json.dumps(records,indent=2)+'\n');assert not any(mismatches.values()),records[-1]
report=dict(status='PASS_NATIVE_ISA_METADATA_V2',device_qualified=False,records=records,library_sha256=hashlib.sha256((build/'libgaudi_route_metadata_v2_tpc.so').read_bytes()).hexdigest())
(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'cases':len(records)}))
