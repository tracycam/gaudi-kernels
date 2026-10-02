"""One bounded module-7 probe, using the committed exported source identity."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--build',required=True);p.add_argument('--timeout',type=int,default=180);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args();root=Path(__file__).resolve().parents[2];identity=json.loads((root/'source-identity.json').read_text());args=a.args[1:] if a.args[:1]==['--'] else a.args
command=[sys.executable,'tools/run_device_probe.py','--module','7','--case',a.case,'--git-commit',identity['git_commit'],'--timeout',str(a.timeout),'--state-timeout','15','--kernel-library',a.build+'/exact/libmxfp4_exact_tpc.so','--kernel-library',a.build+'/history/libmxfp4_overhead_tpc.so','--',a.build+'/probe',*args]
raise SystemExit(subprocess.run(command,cwd=root).returncode)
