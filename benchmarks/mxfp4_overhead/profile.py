"""Capture one complete recipe replay; profiler timings are diagnostic only."""
import os,subprocess,sys
from pathlib import Path
out=Path(os.environ['PROBE_OUT']);trace=out/'trace';trace.mkdir();config=out/'profile.json'
# correctness launch1 + ten warmups => first timed invocation12.
subprocess.run(['hl-prof-config','--gaudi2','--config-filename',str(config),'-e','off','-o',str(trace),'-s','mxfp4-overhead',
 '--phase','multi-enq','-g','12-12','-b','64','--invoc','json','--merged','json,hltv','--trace-analyzer','on','--trace-analyzer-csv','on','--host','on','--add-pid','off'],check=True)
os.environ.update(HABANA_PROFILE='1',HABANA_PROF_CONFIG=str(config))
os.execv(sys.argv[1],sys.argv[1:])
