"""Bounded module7 only, after coordinator release; profile is a separate case."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--profile',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];commit=json.loads((root/'source-identity.json').read_text())['git_commit']
cmd=[sys.executable,'tools/run_device_probe.py','--module','7','--case',a.case,'--git-commit',commit,'--timeout','180']
for f in ('fixtures/executor-a/executor/tpc/libnative_tpc.so','fixtures/executor-a/tpc/libbatch_tpc.so','fixtures/executor-a/precision-tpc/libprecision_tpc.so','runtime/libgaudi_moe_activation_folded_tpc.so','runtime/libgaudi_down_activation_tpc.so','builds/tpc/libgaudi_gp_scale_tail_tpc.so'):cmd+=['--kernel-library',f]
env=[]
if a.profile:
 d=root/'profiles'/a.case;d.mkdir(parents=True,exist_ok=False)
 subprocess.run(['hl-prof-config','--gaudi2','--config-filename',str(d/'profile.json'),'-e','off','-o',str(d/'trace'),'-s',a.case,'--phase','multi-enq','-g','1-1000','-b','128','--invoc','json','--merged','json,hltv','--trace-analyzer','on','--trace-analyzer-csv','on','--host','on','--add-pid','off'],check=True,timeout=20)
 env=['HABANA_PROFILE=1','HABANA_PROF_CONFIG='+str(d/'profile.json')]
cmd+=['--','env',*env,sys.executable,'benchmarks/moe_production_issue/device_gate.py','--executor-fixture','fixtures/executor-a','--helper','runtime/gaudi_view_safe_experiment.so','--helper-sha256','ab96dc607a7523c7c0cfd72869c51fed2d540ffeeea1161a19d9d2fe77527192','--scale-tail-library','builds/torch/gaudi_gp_scale_tail.so','--control-folded-library','runtime/gaudi_moe_activation_folded.so','--fixed-down-library','runtime/gaudi_down_activation.so']
if a.profile:cmd+=['--quick']
raise SystemExit(subprocess.call(cmd,cwd=root))
