"""Run the existing independent CPU/actual-ISA gates through only the bundle."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--core-control',type=Path,required=True);p.add_argument('--tiles-control',type=Path,required=True)
p.add_argument('--old-gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2];commands=[]
def run(command,label,timeout=300):
    commands.append(dict(label=label,command=list(map(str,command))))
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with(out/(label+'.log')).open('w')as f:
        subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
metadata=out/'metadata-build';metadata.mkdir()
shutil.copy2(a.bundle,metadata/'libgaudi_route_metadata_v3_tpc.so')
run(['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',root/'benchmarks/moe_route_metadata_v3/simulator.cpp',
     '-L/usr/lib/habanatools','-Wl,-rpath,/usr/lib/habanatools','-ltpc_tests_core_ext','-ldl','-o',metadata/'simulator'],
    'metadata-compile',45)
run([sys.executable,root/'benchmarks/moe_route_metadata_v3/sim_gate.py','--build',metadata,'--output',out/'metadata'],
    'metadata')
builds=out/'decode-builds';builds.mkdir()
for name,library in [('core-control',a.core_control),('core-hoist',a.bundle),('tiles-control',a.tiles_control),('tiles-hoist',a.bundle)]:
    d=builds/name;d.mkdir();shutil.copy2(library,d/('libmxfp4_moe_graph_tpc.so'if name.startswith('core')else'libgaudi_route_tiles_tpc.so'))
run([sys.executable,root/'benchmarks/route_tile_hoist/isa_gate.py','--builds',builds,'--output',out/'decode'],'decode')
run([sys.executable,root/'benchmarks/moe_route_tiles/isa_gate.py','--library',a.bundle,
     '--old-gate',a.old_gate,'--output',out/'consumers'],'consumers')
reports={name:json.loads((out/name/'result.json').read_text())for name in ('metadata','decode','consumers')}
assert all(r['status'].startswith('PASS')for r in reports.values())
(out/'result.json').write_text(json.dumps(dict(status='PASS_BUNDLE_ACTUAL_ISA_ALL_USED_GUIDS',
    tested_GUIDs=8,core_other_GUIDs='full ELF and all-API differential, not newly functionally qualified',
    reports=reports,device_acquired=False),indent=2)+'\n')
print('PASS_BUNDLE_ACTUAL_ISA_ALL_USED_GUIDS')
