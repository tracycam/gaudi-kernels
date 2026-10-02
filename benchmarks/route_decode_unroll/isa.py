"""Reuse the complete decoded storage-bit oracle on both unroll candidates."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--builds',type=Path,required=True)
p.add_argument('--tiles',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2];results=[]
for factor in (2,4):
    base=a.output/f'u{factor}';base.mkdir();builds=base/'builds';builds.mkdir()
    for name,source in [('core-control',a.builds/'u1'),('core-hoist',a.builds/f'u{factor}'),
                        ('tiles-control',a.tiles),('tiles-hoist',a.tiles)]:
        shutil.copytree(source,builds/name)
    command=[sys.executable,str(root/'benchmarks/route_tile_hoist/isa_gate.py'),
             '--builds',str(builds),'--output',str(base/'sim')]
    (base/'command.json').write_text(json.dumps(command)+'\n')
    with(base/'driver.log').open('w')as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
    result=json.loads((base/'sim/result.json').read_text());assert result['status']=='PASS_ACTUAL_ISA_HOIST'
    results.append(dict(unroll=factor,result=result))
(a.output/'result.json').write_text(json.dumps(dict(status='PASS_BOTH_ACTUAL_ELFS',results=results),indent=2)+'\n')
print('PASS_BOTH_ACTUAL_ELFS')
