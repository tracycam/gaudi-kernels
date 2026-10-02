"""Generate bounded byte oracles, then invoke the native probe inside shared runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from cpu_gate import C,FLOOR,bf16_bits,bf16_float,half_code,ocp_code
def generate(fixture,case,table):
 fixture.mkdir()
 m=128 if case=='mantissas129' else 512 if case=='random512' else 513 if case=='random513' else 1
 k=513 if case=='extreme513' else 2048 if case.startswith('random') else 129
 rng=np.random.default_rng(129)
 if case=='mantissas129':
  maxima=(1+np.arange(128,dtype=np.float32)/128)[:,None]
  x=bf16_float(bf16_bits(rng.uniform(-.9,.9,(m,k))*maxima));x[:,0]=maxima[:,0];x[:,1]=-maxima[:,0];x[:,2]=0.;x[:,3]=-0.
  x[:,4]=bf16_float(bf16_bits(maxima[:,0]*1e-5));x[:,5]=-x[:,4];x[89,6]=np.float32(1.1086463928222656e-05);bits=bf16_bits(x)
 elif case=='clamp129':bits=np.resize(np.array([0,0x8000,1,0x8001,127,0x807f,128,0x8080,0x2edb,0xaedb],np.uint16),(m,k))
 elif case=='extreme513':bits=np.resize(np.array([0,0x8000,1,0x8001,127,0x807f,128,0x8080,0x7f7f,0xff7f,0x3f80,0xbf80],np.uint16),(m,k))
 else:bits=bf16_bits(rng.normal(0,.3,(m,k)))
 x=bf16_float(bits);scale=(np.maximum(np.max(np.abs(x),axis=1,keepdims=True),FLOOR)*C).astype(np.float32)
 expected=half_code(ocp_code((x.astype(np.float64)/scale.astype(np.float64)).astype(np.float32)))
 bits.tofile(fixture/'input.bin');expected.tofile(fixture/'expected-q.bin');(scale*2).tofile(fixture/'expected-scale.bin');(fixture/'table.bin').write_bytes(table.read_bytes())
 return m,k

p=argparse.ArgumentParser();p.add_argument('--binary',required=True,type=Path);p.add_argument('--table',required=True,type=Path)
p.add_argument('--variant',choices=['baseline','amax16','lut_single','lut_single4'],default='lut_single4')
p.add_argument('--case',choices=['mantissas129','clamp129','extreme513','random512','random513'],default='mantissas129')
p.add_argument('--suite',action='store_true');a=p.parse_args();out=Path(os.environ['PROBE_OUT'])
if a.suite:
 plan=[(v,'mantissas129') for v in ['baseline','lut_single','lut_single4']]+[('lut_single4',c) for c in ['clamp129','extreme513']]+[(v,c) for c in ['random512','random513'] for v in ['baseline','lut_single','lut_single4']]
else:plan=[(a.variant,a.case)]
records=[];lines=[]
for variant,case in plan:
 target=out/(variant+'-'+case) if a.suite else out
 if a.suite:target.mkdir()
 fixture=target/'fixture';m,k=generate(fixture,case,a.table)
 metadata={'case':case,'M':m,'K':k,'variant':variant,'bench_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),
  'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture.iterdir()},'reference':'RN32(x / RN32(max(amax, floor) * RN32(1/448))), OCP RNE, native half RNE'}
 (target/'quant-fixture.json').write_text(json.dumps(metadata,indent=2)+'\n');records.append((target,metadata))
 lines.append(f'{variant} {m} {k} {fixture.resolve()} {target.resolve()}')
manifest=out/'batch.txt';manifest.write_text('\n'.join(lines)+'\n')
command=[str(a.binary.resolve()),'batch',str(manifest.resolve())] if a.suite else [str(a.binary.resolve()),a.variant,str(records[0][1]['M']),str(records[0][1]['K']),str((out/'fixture').resolve())]
code=subprocess.run(command).returncode
if a.suite:
 summary=[]
 for target,metadata in records:
  rows=[]
  if (target/'run.log').is_file():
   rows=[json.loads(line) for line in (target/'run.log').read_text().splitlines() if line.startswith('{')]
  metadata['rows']=rows;summary.append(metadata)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
raise SystemExit(code)
