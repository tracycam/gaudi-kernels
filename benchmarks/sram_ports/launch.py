"""Scoped benchmark settings; CSE must not remove repeated MME consumers."""
import os
import sys
import subprocess
from pathlib import Path
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',ENABLE_CSE_OPTIMIZATION='false')
if os.environ.get('GK_PORT_BMON'):
    out=Path(os.environ['PROBE_OUT']);config=out/'ports-bmon.json'
    subprocess.run(['hl-prof-config','--gaudi2','--config-filename',str(config),'-e','off','-o',str(out/'trace'),'-s','ports','--phase','multi-enq','-g','7-7','-b','64','--invoc','json','--merged','json,hltv','--host','on','--trace-analyzer','on','--add-pid','off',
        '--tpc_bmon','[[0,1][bw][2048,1024][]]',
        '--mme_sbte_bmon','[[0][bw][2048,1024][]]',
        '--mme_acc_bmon','[[0,1][bw][2048,1024][]]',
        '--edma_bmon','[[0,1][bw][2048,1024][]]',
        '--tpc_spmu','[[57,65,71,72,74,75][2048,on]]',
        '--mme_sbte_spmu','[[6,7,8,9,10,11][2048,on]]',
        '--mme_acc_spmu','[[0,1][2048,on]]'],check=True)
    os.environ.update(HABANA_PROFILE='1',HABANA_PROF_CONFIG=str(config))
command=sys.argv[1:]
os.execv(command[0],command)
