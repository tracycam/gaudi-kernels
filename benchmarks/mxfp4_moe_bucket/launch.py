"""Optional profiling setup inside the bounded probe child, then exec native."""
import argparse, os, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--profile-launch',type=int);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if a.profile_launch:
 out=Path(os.environ['PROBE_OUT']);trace=out/'trace';trace.mkdir();config=out/'profile.json'
 command=['hl-prof-config','--gaudi2','--config-filename',str(config),'-e','off','-o',str(trace),'-s','mxfp4-bucket','--phase','multi-enq','-g',f'{a.profile_launch}-{a.profile_launch}','-b','64','--invoc','json','--merged','json,hltv','--trace-analyzer','on','--trace-analyzer-csv','on','--host','on','--add-pid','off']
 with (out/'profile-config.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=15)
 os.environ.update(HABANA_PROFILE='1',HABANA_PROF_CONFIG=str(config))
os.execv(cmd[0],cmd)
