"""Bounded two-mode attribution; no profile data is used as latency comparison."""
import json
import os
from pathlib import Path
import subprocess
import sys

out = Path(os.environ['PROBE_OUT'])
trace = out/'trace'
trace.mkdir()
config = out/'profile.json'
command = ['hl-prof-config', '--gaudi2', '--config-filename', str(config), '-e', 'off',
           '-o', str(trace), '-s', 'mac', '--phase', 'multi-enq', '-g', '7-30', '-b', '64',
           '--invoc', 'json', '--merged', 'json,hltv', '--trace-analyzer', 'on',
           '--trace-analyzer-csv', 'on', '--host', 'on', '--add-pid', 'off']
with (out/'config.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30)
(out/'profile-command.json').write_text(json.dumps({'config_command': command,
    'command': sys.argv[1:], 'scope': 'engine attribution, not unprofiled latency'}, indent=2)+'\n')
env = dict(os.environ, HABANA_PROFILE='1', HABANA_PROF_CONFIG=str(config))
raise SystemExit(subprocess.call(sys.argv[1:], env=env))
