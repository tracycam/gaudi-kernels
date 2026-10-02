"""Require exactly one external TPC database before the existing full graph gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--bundle-library',type=Path,required=True)
a,remaining=p.parse_known_args()
bundle=a.bundle_library.resolve()
paths=[Path(v).resolve()for v in os.environ.get('GC_KERNEL_PATH','').split(':')if v]
assert paths==[Path('/usr/lib/habanalabs/libtpc_kernels.so').resolve(),bundle],paths
record=dict(GC_KERNEL_PATH=os.environ['GC_KERNEL_PATH'],external_TPC_libraries=1,
            bundle_path=str(bundle),bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
            original_three_TPC_libraries_registered=False)
out=Path(os.environ['PROBE_OUT']);(out/'bundle-binding.json').write_text(json.dumps(record,indent=2)+'\n')
driver=Path(__file__).resolve().parents[1]/'route_decode_unroll/device.py'
sys.argv=[str(driver),*remaining]
runpy.run_path(str(driver),run_name='__main__')
result=json.loads((out/'result.json').read_text());result['bundle_binding']=record
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
