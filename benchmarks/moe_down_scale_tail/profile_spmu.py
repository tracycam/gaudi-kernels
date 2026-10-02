"""Process-local M1 stall witness, retaining raw counter units and exact config."""
import json,os,subprocess,sys
from pathlib import Path
out=Path(os.environ['PROBE_OUT']);trace=out/'trace';trace.mkdir();config=out/'profile.json'
selection='[[10,11,12,17,28,57][256,on]]'
command=['hl-prof-config','--gaudi2','--config-filename',str(config),'-e','off','-o',str(trace),
 '-s','mac','--phase','multi-enq','-g','0-300','-b','64','--invoc','json','--merged','json,hltv',
 '--trace-analyzer','on','--trace-analyzer-csv','on','--host','on','--add-pid','off','--tpc_spmu',selection]
with(out/'config.log').open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
(out/'profile-command.json').write_text(json.dumps(dict(config_command=command,payload=sys.argv[1:],
 spmu_selection=selection,requested_sample_rate_cycles=256,clear_on_trace_dump=True,
 scope='Diagnostic only. Decoded raw samples may overlap; no additive cycles, bandwidth or current-clock inference.'),indent=2)+'\n')
raise SystemExit(subprocess.call(sys.argv[1:],env=dict(os.environ,HABANA_PROFILE='1',HABANA_PROF_CONFIG=str(config))))
