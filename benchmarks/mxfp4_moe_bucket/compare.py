"""Same-fixture native bucket vs CAP=T output bits; separate recipes, no extra MME."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--bucket',type=Path,required=True);p.add_argument('--capacity-T',dest='reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
for case in [a.bucket,a.reference]:assert json.loads((case/'exit.json').read_text())['runner_exit_code']==0
for name in ['packed.bin','scales.bin','activation.bin','ids.bin','routing.bin']:
 assert hashlib.sha256((a.bucket/name).read_bytes()).digest()==hashlib.sha256((a.reference/name).read_bytes()).digest(),name
fixture=json.loads((a.bucket/'fixture.json').read_text());gate=fixture['mode']=='gate';dtype=np.uint16 if gate else np.uint32
got=np.fromfile(a.bucket/'output.bin',dtype);ref=np.fromfile(a.reference/'output.bin',dtype);assert got.shape==ref.shape
bad=int(np.count_nonzero(got!=ref));nonfinite=int(np.count_nonzero((got&0x7f80)==0x7f80)) if gate else int(np.count_nonzero(~np.isfinite(got.view(np.float32))))
result={'scope':'same original bytes/activation/routes, separate native recipes','checked':int(got.size),'bit_mismatches':bad,'nonfinite':nonfinite,'all_pass':bad==0 and nonfinite==0}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(0 if result['all_pass'] else 3)
