"""Bounded physical trace of both MoE graphs; name must not shadow stdlib profile."""
import os
from pathlib import Path
import subprocess
import sys
out = Path(os.environ['PROBE_OUT'])
trace = out/'trace'
trace.mkdir()
config = out/'profile.json'
subprocess.run(['hl-prof-config', '--gaudi2', '--config-filename', str(config), '-e', 'off', '-o', str(trace),
                '-s', 'moe_prefetch', '--phase', 'multi-enq', '-g', '1-80', '-b', '128', '--invoc', 'json',
                '--merged', 'json,hltv', '--trace-analyzer', 'on', '--trace-analyzer-csv', 'on',
                '--host', 'on', '--add-pid', 'off'], check=True)
os.environ.update(HABANA_PROFILE='1', HABANA_PROF_CONFIG=str(config))
os.execv(sys.argv[1], sys.argv[1:])
