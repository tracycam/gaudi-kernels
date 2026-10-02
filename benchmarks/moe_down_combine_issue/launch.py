"""Launch only after coordinator grants module7; all device state via shared runner."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--manifest',type=Path,default=Path('manifest.json'));p.add_argument('--variants',nargs='+',choices=('p6','p4'),default=['p6']);p.add_argument('--profile',action='store_true');p.add_argument('--streaming',action='store_true');p.add_argument('--timing',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[2]
manifest=json.loads((root/a.manifest).read_text());identity=json.loads((root/'source-identity.json').read_text());assert manifest['source_commit']==identity['git_commit']
keys=manifest['kernel_library_keys'];assert len(keys)+1<=15
cmd=[sys.executable,'tools/run_device_probe.py','--module','7','--case',a.case,'--git-commit',identity['git_commit'],'--timeout','180']
for key in keys:cmd+=['--kernel-library',manifest['files'][key]['path']]
env=[]
if a.profile:
    folder=root/'profiles'/a.case;folder.mkdir(parents=True,exist_ok=False)
    subprocess.run(['hl-prof-config','--gaudi2','--config-filename',str(folder/'profile.json'),'-e','off','-o',str(folder/'trace'),'-s',a.case,'--phase','multi-enq','-g','1-4000','-b','256','--invoc','json','--merged','json,hltv','--trace-analyzer','on','--trace-analyzer-csv','on','--host','on','--add-pid','off'],check=True,timeout=20)
    env=['HABANA_PROFILE=1','HABANA_PROF_CONFIG='+str(folder/'profile.json')]
cmd+=['--','env',*env,sys.executable,'benchmarks/moe_down_combine_issue/device_gate.py','--manifest',str(a.manifest),'--variants',*a.variants]
if a.profile:cmd+=['--profile']
if a.streaming:cmd+=['--streaming']
if a.timing or a.profile:cmd+=['--timing']
raise SystemExit(subprocess.call(cmd,cwd=root))
