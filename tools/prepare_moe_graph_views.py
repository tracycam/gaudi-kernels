"""Emit a patched sitecustomize and unified diff into a NEW directory only."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'python'))
from gaudi_kernels.moe_graph_views import rewrite_sitecustomize_source
p = argparse.ArgumentParser()
p.add_argument('--sitecustomize', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
source = a.sitecustomize.read_text()
rewritten = rewrite_sitecustomize_source(source)
a.output_dir.mkdir(parents=True, exist_ok=False)
(a.output_dir/'sitecustomize.py').write_text(rewritten)
(a.output_dir/'sitecustomize.patch').write_text(''.join(difflib.unified_diff(
    source.splitlines(True), rewritten.splitlines(True), fromfile='a/executor/sitecustomize.py',
    tofile='b/executor/sitecustomize.py')))
(a.output_dir/'manifest.json').write_text(json.dumps({
    'source': str(a.sitecustomize.resolve()), 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'patched_sha256': hashlib.sha256(rewritten.encode()).hexdigest(),
    'status': 'OFFLINE_CANDIDATE', 'active_executor_modified': False}, indent=2)+'\n')
print(a.output_dir)
