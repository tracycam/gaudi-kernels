"""Configure scoped bus monitors, then execute the assigned probe only."""
import os
from pathlib import Path
import subprocess
import sys

out = Path(os.environ['PROBE_OUT'])
config = out/'bmon.json'
subprocess.run(['hl-prof-config', '--gaudi2', '--config-filename', str(config),
                '-e', 'off', '-o', str(out/'trace'), '-s', 'contention',
                '--phase', 'multi-enq', '-g', '7-7', '-b', '64',
                '--invoc', 'json', '--merged', 'json,hltv', '--host', 'on',
                '--trace-analyzer', 'on', '--add-pid', 'off',
                '--tpc_bmon', '[[0,1][bw,lat,out][512,256][]]',
                '--mme_sbte_bmon', '[[0][bw,lat,out][512,256][]]'], check=True)
os.environ.update(HABANA_PROFILE='1', HABANA_PROF_CONFIG=str(config))
command = sys.argv[1:]
if command[:1] == ['--']:
    command = command[1:]
os.execv(command[0], command)
