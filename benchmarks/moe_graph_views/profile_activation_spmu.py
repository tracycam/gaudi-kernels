"""One local-config diagnostic profile; counter categories may overlap."""
import json
import os
from pathlib import Path
import subprocess
import sys

out=Path(os.environ['PROBE_OUT']);trace=out/'trace';trace.mkdir();config=out/'profile.json'
selection='[[10,11,12,17,21,28][1024,on]]'
command=['hl-prof-config','--gaudi2','--config-filename',str(config),'-e','off','-o',str(trace),
         '-s','mac','--phase','multi-enq','-g','7-30','-b','64','--invoc','json','--merged','json,hltv',
         '--trace-analyzer','on','--trace-analyzer-csv','on','--host','on','--add-pid','off','--tpc_spmu',selection]
with (out/'config.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
(out/'profile-command.json').write_text(json.dumps({'config_command':command,'command':sys.argv[1:],
    'scope':'diagnostic raw hardware C events only; overlapping stall counters are not additive time',
    'spmu_selection':selection,'sample_rate_cycles':1024,'clear_on_trace_dump':True,
    'counter_names':{'10':'STALL_ON_DCACHE_MISS','11':'STALL_ON_POP_FROM_SB','12':'STALL_ON_LOOKUP_CACHE_MISS',
                     '17':'STALL_SPU_ANY','21':'STALL','28':'VECTOR_PIPE_EXEC'}},indent=2)+'\n')
raise SystemExit(subprocess.call(sys.argv[1:],env=dict(os.environ,HABANA_PROFILE='1',HABANA_PROF_CONFIG=str(config))))
