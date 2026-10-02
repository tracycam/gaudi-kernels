"""Construct the owned module5 launch; no device action with --dry-run."""
import argparse,json,subprocess,sys
from pathlib import Path
from plan import validate_runtime
p=argparse.ArgumentParser();p.add_argument('--case',required=True)
for n in('input','runtime-fixture','core-torch','metadata-torch','tiles-torch','bundle-tpc','wide-tpc','wide-torch','library-pins'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--tokens',type=int,choices=(512,513),default=512);p.add_argument('--timeout',type=int,default=900);p.add_argument('--timing-anchor',choices=('broadcast','c32'),default='broadcast');p.add_argument('--trials',type=int,default=3);p.add_argument('--routing-pattern',choices=('checkpoint','uniform_all'),default='checkpoint');p.add_argument('--dry-run',action='store_true');a=p.parse_args();assert 60<=a.timeout<=900
root=Path(__file__).resolve().parents[2];_,libs=validate_runtime(a.runtime_fixture);identity=root/'source-identity.json';commit=json.loads(identity.read_text())['git_commit']if identity.exists()and not(root/'.git').exists()else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
def relative(p):return str(Path(p).resolve().relative_to(root))
cmd=[sys.executable,'tools/run_device_probe.py','--module','5','--case',a.case,'--git-commit',commit,'--timeout',str(a.timeout)]
for path in [a.runtime_fixture/v for v in libs]+[a.bundle_tpc,a.wide_tpc]:cmd+=['--kernel-library',relative(path)]
cmd+=['--',sys.executable,'benchmarks/moe_route_wide/device_chain.py','--tokens',str(a.tokens),'--timing-anchor',a.timing_anchor,'--trials',str(a.trials),'--routing-pattern',a.routing_pattern]
for name in('input','runtime_fixture','core_torch','metadata_torch','tiles_torch','bundle_tpc','wide_tpc','wide_torch','library_pins'):cmd+=['--'+name.replace('_','-'),relative(getattr(a,name))]
print(json.dumps(dict(source_commit=commit,device_run=not a.dry_run,command=cmd),indent=2),flush=True)
if not a.dry_run:raise SystemExit(subprocess.call(cmd,cwd=root))
