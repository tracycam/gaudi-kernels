"""Process-local complete-MoE native profile; no global profiler modification."""
import json
import os
from pathlib import Path
import subprocess
import sys
out=Path(os.environ['PROBE_OUT']);trace=out/'trace';trace.mkdir();config=out/'profile.json'
command=['hl-prof-config','--gaudi2','--config-filename',str(config),'-e','off','-o',str(trace),'-s','mac',
         '--phase','multi-enq','-g','0-300','-b','64','--invoc','json','--merged','json,hltv',
         '--trace-analyzer','on','--trace-analyzer-csv','on','--host','on','--add-pid','off']
with(out/'config.log').open('w')as log:subprocess.run(command,stdout=log,stderr=log,check=True,timeout=30)
(out/'profile-command.json').write_text(json.dumps({'config':command,'payload':sys.argv[1:]},indent=2)+'\n')
raise SystemExit(subprocess.call(sys.argv[1:],env=dict(os.environ,HABANA_PROFILE='1',HABANA_PROF_CONFIG=str(config))))
