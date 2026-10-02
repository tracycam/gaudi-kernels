import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--build',required=True);p.add_argument('--timeout',type=int,default=180);p.add_argument('--profile',action='store_true');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args();root=Path(__file__).resolve().parents[2];commit=json.loads((root/'source-identity.json').read_text())['git_commit'];args=a.args[1:] if a.args[:1]==['--'] else a.args;out=root/'results'/a.case
env=dict(os.environ,ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),SRAM_SLICER_GRAPH_VISUALIZATION='1',DUMP_POST_GRAPHS=str(out/'post_graph.json'))
command=[a.build+'/probe',*args]
if a.profile:
 # GP correctness has one launch. Linear: correctness1+warmup10, first timing12.
 command=[sys.executable,'benchmarks/mxfp4_moe_bucket/launch.py','--profile-launch',str(1 if len(args)>1 and args[1]=='gate' else 12),'--',*command]
raise SystemExit(subprocess.run([sys.executable,'tools/run_device_probe.py','--module','7','--case',a.case,'--git-commit',commit,'--state-timeout','15','--timeout',str(a.timeout),'--kernel-library',a.build+'/libmxfp4_moe_bucket_tpc.so','--',*command],cwd=root,env=env).returncode)
