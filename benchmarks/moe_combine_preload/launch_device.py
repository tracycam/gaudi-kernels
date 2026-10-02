"""Bounded module7 invocation; --dry-run validates bindings without HPU use."""
import argparse,json,subprocess,sys
from pathlib import Path
from plan import validate,validate_combine
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--runtime-fixture',type=Path,required=True);p.add_argument('--combine-tpc',type=Path,required=True);p.add_argument('--combine-torch',type=Path,required=True);p.add_argument('--combine-torch-sha256',required=True);p.add_argument('--rows',default='1,2,8');p.add_argument('--replays',type=int,default=16);p.add_argument('--trials',type=int,default=3);p.add_argument('--timeout',type=int,default=180);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];plan=validate(a.runtime_fixture);binding=validate_combine(a.combine_tpc);identity=root/'source-identity.json'
commit=json.loads(identity.read_text())['git_commit']if identity.exists()and not(root/'.git').exists()else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
assert 30<=a.timeout<=900
# The runner hashes registered libraries inside the committed/exported tree.
def relative(path):return str(Path(path).resolve().relative_to(root))
cmd=[sys.executable,'tools/run_device_probe.py','--module','7','--case',a.case,'--git-commit',commit,'--timeout',str(a.timeout)]
for key in plan['tpc_order']:cmd+=['--kernel-library',relative(a.runtime_fixture/plan['files'][key]['path'])]
cmd+=['--kernel-library',relative(a.combine_tpc),'--',sys.executable,'benchmarks/moe_combine_preload/device_chain.py','--runtime-fixture',relative(a.runtime_fixture),'--combine-tpc',relative(a.combine_tpc),'--combine-torch',relative(a.combine_torch),'--combine-torch-sha256',a.combine_torch_sha256,'--rows',a.rows,'--replays',str(a.replays),'--trials',str(a.trials)]
print(json.dumps(dict(source_commit=commit,command=cmd,combine_binding=binding,device_run=not a.dry_run),indent=2),flush=True)
if not a.dry_run:raise SystemExit(subprocess.call(cmd,cwd=root))
