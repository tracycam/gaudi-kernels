"""Per-process graph dumps. Must be invoked inside run_device_probe.py."""
import os
from pathlib import Path
import subprocess,sys
out=Path(os.environ['PROBE_OUT'])
env=dict(os.environ,ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),
 GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(out/'graphs'),SRAM_SLICER_GRAPH_VISUALIZATION='1')
(out/'graphs').mkdir(exist_ok=True)
raise SystemExit(subprocess.call(sys.argv[1:],env=env))
