import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--build',required=True);p.add_argument('--timeout',type=int,default=180);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args();root=Path(__file__).resolve().parents[2];commit=json.loads((root/'source-identity.json').read_text())['git_commit'];args=a.args[1:] if a.args[:1]==['--'] else a.args;out=root/'results'/a.case
env=dict(os.environ,ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),SRAM_SLICER_GRAPH_VISUALIZATION='1',DUMP_POST_GRAPHS=str(out/'post_graph.json'))
raise SystemExit(subprocess.run([sys.executable,'tools/run_device_probe.py','--module','7','--case',a.case,'--git-commit',commit,'--state-timeout','15','--timeout',str(a.timeout),'--kernel-library',a.build+'/libmxfp4_moe_grouped_tpc.so','--',a.build+'/probe',*args],cwd=root,env=env).returncode)
