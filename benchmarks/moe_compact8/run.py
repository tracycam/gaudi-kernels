"""Launch only the isolated module6 bounded gate, with prebuilt original DLLs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser();p.add_argument('--runtime-fixture',type=Path,required=True)
p.add_argument('--weights-fixture',type=Path,required=True);p.add_argument('--case',required=True)
p.add_argument('--profile',action='store_true');p.add_argument('--timeout',type=int,default=180);a=p.parse_args()
root=Path(__file__).resolve().parents[2];identity=json.loads((root/'source-identity.json').read_text())
runtime=a.runtime_fixture.resolve(strict=True);weights=a.weights_fixture.resolve(strict=True)
assert runtime.is_relative_to(root)and weights.is_relative_to(root)
plan=json.loads((runtime/'plan.json').read_text());assert 1<=a.timeout<=180
command=[sys.executable,str(root/'tools/run_device_probe.py'),'--module','6','--case',a.case,
         '--git-commit',identity['git_commit'],'--timeout',str(a.timeout)]
for name in plan['tpc_libraries']:command+=['--kernel-library',str((runtime/name).relative_to(root))]
command+=['--']
if a.profile:command+=[sys.executable,str(root/'benchmarks/moe_compact8/capture_profile.py')]
command+=[sys.executable,str(root/'benchmarks/moe_compact8/device.py'),
          '--runtime-fixture',str(runtime),'--weights-fixture',str(weights)]
if a.profile:command+=['--profile-only']
raise SystemExit(subprocess.call(command,cwd=root))
