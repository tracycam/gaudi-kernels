"""Run a unique bounded case using the exported, verified Git source identity."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--build',required=True);p.add_argument('--module',type=int,choices=[7],required=True);p.add_argument('--profile',action='store_true');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
root=Path(__file__).resolve().parents[2];identity=json.loads((root/'source-identity.json').read_text());args=a.args[1:] if a.args[:1]==['--'] else a.args
cmd=[sys.executable,str(root/'tools/run_device_probe.py'),'--module',str(a.module),'--case',a.case,'--git-commit',identity['git_commit'],'--timeout','900',
 '--kernel-library',a.build+'/libmxfp4_overhead_tpc.so','--kernel-library',a.build+'/baseline/libgaudi_mxfp4_linear_tpc.so','--']
if a.profile:cmd += [sys.executable,'benchmarks/mxfp4_overhead/profile.py']
cmd += [a.build+'/overhead_probe',*args]
raise SystemExit(subprocess.run(cmd,cwd=root).returncode)
