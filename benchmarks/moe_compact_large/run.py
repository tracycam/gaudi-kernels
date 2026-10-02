"""Explicit single-module large-M probe; never alters production row guards."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('runtime-fixture','weights-fixture','scale-torch','scale-tpc'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--tokens',type=int,choices=(32,128,512,513),required=True)
p.add_argument('--compact-kind',choices=('vector','scalar'),default='vector')
p.add_argument('--case',required=True);p.add_argument('--profile',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];identity=json.loads((root/'source-identity.json').read_text())
runtime=a.runtime_fixture.resolve(strict=True);weights=a.weights_fixture.resolve(strict=True)
assert runtime.is_relative_to(root)and weights.is_relative_to(root)
plan=json.loads((runtime/'plan.json').read_text())
command=[sys.executable,str(root/'tools/run_device_probe.py'),'--module','6','--case',a.case,'--git-commit',identity['git_commit'],'--timeout','180']
for name in plan['tpc_libraries']:command+=['--kernel-library',str((runtime/name).relative_to(root))]
scale=a.scale_tpc.resolve(strict=True);assert scale.is_relative_to(root)
command+=['--kernel-library',str(scale.relative_to(root)),'--']
if a.profile:command+=[sys.executable,str(root/'benchmarks/moe_compact8/capture_profile.py')]
command+=[sys.executable,str(root/'benchmarks/moe_compact_large/device.py'),'--runtime-fixture',str(runtime),'--weights-fixture',str(weights),'--tokens',str(a.tokens),'--scale-torch',str(a.scale_torch.resolve(strict=True)),'--scale-tpc',str(scale)]
if a.profile:command+=['--profile-only']
command+=['--compact-kind',a.compact_kind]
raise SystemExit(subprocess.call(command,cwd=root))
