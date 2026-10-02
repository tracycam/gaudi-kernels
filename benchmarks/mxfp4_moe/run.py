import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--build',required=True);p.add_argument('--timeout',type=int,default=180);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args();root=Path(__file__).resolve().parents[2];commit=json.loads((root/'source-identity.json').read_text())['git_commit'];args=a.args[1:] if a.args[:1]==['--'] else a.args
raise SystemExit(subprocess.run([sys.executable,'tools/run_device_probe.py','--module','7','--case',a.case,'--git-commit',commit,'--state-timeout','15','--timeout',str(a.timeout),'--kernel-library',a.build+'/libmxfp4_moe_tpc.so','--',a.build+'/probe',*args],cwd=root).returncode)
