"""One bounded module7 runner entry. Do not invoke before coordinator handoff."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--fixtures',required=True);p.add_argument('--torch-library',required=True);p.add_argument('--old-torch-library',required=True);p.add_argument('--block-torch-library',required=True);p.add_argument('--counter-library',required=True);p.add_argument('--tpc-library',action='append',required=True);p.add_argument('--phase',choices=['reductions','chain','all'],default='all');p.add_argument('--timeout',type=int,default=180);p.add_argument('--profile',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];assert 1<=a.timeout<=180
identity=root/'source-identity.json'
if identity.exists():commit=json.loads(identity.read_text())['git_commit']
else:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
cmd=[a.python,str(root/'tools/run_device_probe.py'),'--module','7','--case',a.case,'--git-commit',commit,'--timeout',str(a.timeout)]
for library in a.tpc_library:cmd+=['--kernel-library',library]
env=[]
if a.profile:
 assert a.phase=='chain','profile only the bounded actual QKV chain'
 profile=root/'artifacts/builds/reduce-neumaier-isa-public/profiles'/a.case;profile.mkdir(parents=True,exist_ok=False)
 subprocess.run(['hl-prof-config','--gaudi2','--config-filename',str(profile/'profile.json'),'-e','off','-o',str(profile/'trace'),'-s',a.case,'--phase','multi-enq','-g','1-20','-b','64','--invoc','json','--merged','json,hltv','--trace-analyzer','on','--trace-analyzer-csv','on','--host','on','--add-pid','off'],check=True,timeout=20)
 env=['HABANA_PROFILE=1','HABANA_PROF_CONFIG='+str(profile/'profile.json')]
# run_device_probe deliberately clears inherited LD_PRELOAD. env sets only this
# owned subprocess's read-only launch counter, after its lock/preflight.
cmd+=['--','env',*env,'LD_PRELOAD='+str((root/a.counter_library).resolve()),a.python,'benchmarks/reduce_neumaier_isa/device_public.py','--fixtures',a.fixtures,'--library',a.torch_library,'--old-library',a.old_torch_library,'--block-library',a.block_torch_library,'--phase',a.phase]
if a.profile:cmd+=['--chain-case','qkv-measured-m1','--no-timing','--repeats','2','--trials','1']
raise SystemExit(subprocess.call(cmd,cwd=root))
