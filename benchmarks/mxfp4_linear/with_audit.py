"""Enable per-probe compiler placement artifacts without changing others' env."""
import os, sys
from pathlib import Path
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',
    GRAPH_VISUALIZATION_DIR=str(out/'graphs'),SRAM_SLICER_GRAPH_VISUALIZATION='1',
    DUMP_POST_GRAPHS=str(out/'post_graph.json'))
os.execv(sys.argv[1],sys.argv[1:])
