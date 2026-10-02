"""One bounded module7 diagnostic. Invoke only after explicit coordinator handoff."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--case', required=True)
p.add_argument('--assets', default='assets')
p.add_argument('--python', default=sys.executable)
p.add_argument('--timeout', type=int, default=180)
a = p.parse_args()
assert 1 <= a.timeout <= 180
root = Path(__file__).resolve().parents[2]
commit = json.loads((root / 'source-identity.json').read_text())['git_commit']
assets = (root / a.assets).resolve()
assert assets.is_relative_to(root)
cmd = [a.python, str(root / 'tools/run_device_probe.py'), '--module', '7', '--case', a.case,
       '--git-commit', commit, '--timeout', str(a.timeout)]
for name in ('old-tpc.so', 'hand-tpc.so', 'block-tpc.so'):
    cmd += ['--kernel-library', str((assets / name).relative_to(root))]
cmd += ['--', 'env', 'LD_PRELOAD=' + str(assets / 'counter.so'), a.python,
        'benchmarks/reduce_neumaier_isa/device_placement.py', '--assets', str(assets)]
code = subprocess.call(cmd, cwd=root)
if code:
    raise SystemExit(code)
raise SystemExit(subprocess.call([a.python, 'benchmarks/reduce_neumaier_isa/audit_placement_probe.py',
                                 str(root / 'results' / a.case / 'placement')], cwd=root))
