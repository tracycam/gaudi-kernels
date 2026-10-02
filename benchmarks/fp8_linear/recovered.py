"""Same-fixture W8A8 comparison, one device acquisition for the selected finite grid."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import numpy as np
from run import fixture,bf16_float
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--table',type=Path,required=True)
p.add_argument('--grid',choices=['main','edges','stream'],required=True)
p.add_argument('--variants',nargs='+',choices=['baseline','lut_single','lut_single4','amax16'],default=['baseline','lut_single','lut_single4'])
a=p.parse_args();out=Path(os.environ['PROBE_OUT']);binary=a.build.resolve()/'fp8_linear_bench'
shapes={'main':[(m,1024,2048,'random') for m in [1,64,512,513]],
 'edges':[(2,255,513,'range'),(8,257,2049,'ties'),(8,257,769,'cancellation'),(1,135,255,'zero')],
 'stream':[(m,8192,8192,'random') for m in [1,64]]}[a.grid]
records=[];lines=[]
for m,n,k,kind in shapes:
 f=out/f'fixture-{m}-{n}-{k}-{kind}';fixture(f,m,n,k,kind)
 for variant in a.variants:
  target=out/f'{variant}-{m}-{n}-{k}-{kind}';target.mkdir()
  records.append({'M':m,'N':n,'K':k,'kind':kind,'variant':variant,'fixture':str(f),'case':str(target),
   'fixture_sha256':json.loads((f/'fixture.json').read_text())['sha256'],
   'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()})
  lines.append(f'w8a8 {m} {n} {k} {f.resolve()} {target.resolve()} {variant}')
manifest=out/'batch.txt';manifest.write_text('\n'.join(lines)+'\n')
graphs=out/'graphs';graphs.mkdir()
env=dict(os.environ,GK_FP8_QUANT_FAST='1',GK_FP8_EPILOGUE_ROWS='1',GK_FP8_RECIPROCAL_TABLE=str(a.table.resolve()),
 ENABLE_EXPERIMENTAL_FLAGS='true',DUMP_POST_GRAPHS=str(out/'post_graph.json'),GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(graphs),SRAM_SLICER_GRAPH_VISUALIZATION='1')
command=[str(binary),'batch',str(manifest.resolve())]
(out/'batch-command.json').write_text(json.dumps({'command':command,'controls':{k:v for k,v in env.items() if k.startswith('GK_')}},indent=2)+'\n')
code=subprocess.run(command,env=env).returncode
for r in records:
 target=Path(r['case']);f=Path(r['fixture']);rows=[]
 if (target/'run.log').is_file():
  rows=[json.loads(line) for line in (target/'run.log').read_text().splitlines() if line.startswith('{')]
 r['rows']=rows
 actual_path=target/'output.bin'
 if actual_path.exists():
  raw=np.fromfile(actual_path,np.uint16);actual=bf16_float(raw).astype(np.float64).reshape(r['M'],r['N'])
  r['output_sha256']=hashlib.sha256(actual_path.read_bytes()).hexdigest();r['full_output_metrics']={}
  for name in ['reference_fp64','gpu_style_w8a8_fp64','adapted_w8a8_fp64']:
   ref=np.fromfile(f/(name+'.bin'),np.float64).reshape(actual.shape)
   r['full_output_metrics'][name]={'relative_l2':float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-30)),'max_abs':float(np.max(np.abs(actual-ref)))}
  baseline=out/f"baseline-{r['M']}-{r['N']}-{r['K']}-{r['kind']}"/'output.bin'
  if baseline.exists():r['changed_bf16_outputs_vs_baseline']=int(np.count_nonzero(raw!=np.fromfile(baseline,np.uint16)))
 times=[row for row in rows if row.get('stage')=='timing'];checks=[row for row in rows if row.get('stage')=='correctness']
 r['passed']=bool(checks and checks[0]['pass'] and len(times)==5)
 if times:
  us=statistics.median(t['event_us'] for t in times);r.update(event_median_us=us,wall_median_us=statistics.median(t['wall_us'] for t in times),
   effective_tflops=2*r['M']*r['N']*r['K']/us/1e6,logical_weight_payload_gb_s=r['N']*r['K']/us/1000)
 (target/'result.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({k:v for k,v in r.items() if k not in ['rows','fixture_sha256','full_output_metrics']}),flush=True)
(out/'summary.json').write_text(json.dumps({'returncode':code,'scope':'complete quantization + native FP8 MME + scale/bias/BF16; excludes one-time weight/table uploads','logical_payload_denominator':'one byte per N*K original-bitwidth prepared weight; no physical-bus claim','results':records},indent=2)+'\n')
raise SystemExit(code if code else 0 if all(r['passed'] for r in records) else 3)
