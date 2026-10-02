"""Repeat certified exact-GP rows, preserving original weights and oracle bits."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,required=True);a=p.parse_args()
assert 1<=a.tokens<=513
source=a.input.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
meta=json.loads((source/'fixture.json').read_text());assert meta['E']==meta['R']==2 and meta['case']=='mixed' and meta['GP_full_FP64_exactly_representable_in_FP32']
rows=np.arange(a.tokens)%meta['T']
repeat={'x','ids','routing','expected','absolute','gp_exact','gate_bits','down_exact'}
provenance={}
for path in sorted(source.glob('*.npy')):
    provenance[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stem in repeat:np.save(out/path.name,np.load(path)[rows])
    else:os.link(path,out/path.name)
meta.update(T=a.tokens,source_fixture=str(source),source_files_sha256=provenance,
    row_repetition_indices=rows.tolist(),device_validated=False,
    scope='Repeated exact original rows for high expert occupancy; does not emulate E384 routing diversity')
(out/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(dict(T=a.tokens,E=2,R=2,out=str(out))))
