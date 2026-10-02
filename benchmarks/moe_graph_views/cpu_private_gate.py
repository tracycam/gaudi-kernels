"""CPU cast/flag/source gate for the private MoE boundary; no bridge is loaded."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import torch
from cpu_gate import function, expect_error
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'python'))
from gaudi_kernels import moe_private_boundary as boundary
p = argparse.ArgumentParser()
p.add_argument('--batch', type=Path, required=True)
p.add_argument('--precision', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
os.environ.update(PT_HPU_LAZY_MODE='1', PT_HPU_LAZY_ACC_PAR_MODE='0', PT_ENABLE_INT64_SUPPORT='0')
lib = torch.library.Library('gaudi_view_safe', 'DEF')
lib.define('normalize_list(Tensor[] inputs) -> Tensor[]')
lib.impl('normalize_list', lambda items: items, 'CPU')
boundary._pin = {'scope': 'CPU stub, not real ABI qualification'}
batch = a.batch.read_text()
ns = {'torch': torch, 'os': os}
rewritten = boundary.rewrite_batch_source(batch, ns)
before = function(batch, 'moe', {'torch': torch, 'os': os}, boundary=True)
after = function(rewritten, 'moe', ns, boundary=True)
oldclone = torch.Tensor.clone
count = [0]
def clone(t, *args, **kwargs):
    count[0] += 1
    return oldclone(t, *args, **kwargs)
rows = []
try:
    torch.Tensor.clone = clone
    for mode in ('compact', 'broadcast'):
        for precise in (False, True):
            os.environ['UNIFIED_PRECISION_ROUTER'] = str(int(precise))
            x = torch.randn(2, 6144).bfloat16()
            ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
            routes = torch.tensor([[.1234567, .8765433], [.25, .75]])
            args = (x, ids, routes, None, None, None, None, None, None)
            count[0] = 0
            base = before(*args, mode=mode)
            assert count[0] == 3
            for flag in ('0', '1'):
                os.environ['GK_MOE_PRIVATE_VIEWS'] = flag
                count[0] = 0
                candidate = after(*args, mode=mode)
                assert count[0] == (3 if flag == '0' else 0)
                assert all(torch.equal(x, y) and x.dtype == y.dtype for x, y in zip(base, candidate))
                rows.append(dict(mode=mode, precise=precise, flag=flag, clones=count[0]))
    precision = a.precision.read_text()
    class Proxy:
        def __getattr__(self, name): return getattr(torch, name)
        ops = types.SimpleNamespace(precision_fix=types.SimpleNamespace(combine=lambda p, r, d, h: r))
    ns = {'torch': Proxy()}
    new_precision = boundary.rewrite_precision_source(precision, ns)
    combine = function(new_precision, 'combine', ns)
    for dtype in (torch.float32, torch.bfloat16):
        route = torch.tensor([[.1234567, .8765433]], dtype=dtype)
        for flag in ('0', '1'):
            os.environ['GK_MOE_PRIVATE_COMBINE'] = flag
            count[0] = 0
            actual = combine(None, route, None, 6144)
            assert count[0] == (1 if flag == '0' else 0)
            assert actual.dtype == torch.float32 and torch.equal(actual, route.float())
            rows.append(dict(combine=True, input_dtype=str(dtype), flag=flag, clones=count[0]))
finally:
    torch.Tensor.clone = oldclone
errors = [expect_error(lambda: boundary.rewrite_batch_source(rewritten, {}), 'three-clone'),
          expect_error(lambda: boundary.rewrite_precision_source(new_precision, {}), 'combine clone')]
errors.append(expect_error(lambda: boundary.install(a.batch, '0'*64), 'fingerprint mismatch'))
errors.append(expect_error(lambda: boundary.prepare(x, ids, routes, True, 'sorted'), 'compact/broadcast'))
boundary._pin = None
errors.append(expect_error(lambda: boundary.normalize_inputs([x]), 'install the exact'))
result = {'status': 'PASS_CPU_ONLY', 'cases': rows, 'rejections': errors,
          'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.batch, a.precision)},
          'scope': 'CPU stubs check casts/flags; separate real ABI/device tests required'}
(a.output_dir/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'status': result['status'], 'cases': len(rows), 'rejections': len(errors)}))
