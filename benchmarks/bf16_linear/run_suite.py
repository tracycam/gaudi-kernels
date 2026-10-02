"""Full-output FP64 checks and complete-recipe timings on the assigned module."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
root=Path(__file__).resolve().parents[2];out=Path(os.environ['PROBE_OUT'])
build=root/os.environ.get('GK_BF16_BUILD','artifacts/builds/bf16-coverage/v1');binary=build/'bf16_linear_benchmark'
phase=os.environ.get('GK_BF16_PHASE','smoke')
fixtures=out/'fixtures';fixtures.mkdir();records=[]
def bf16(v):
 u=np.asarray(v,dtype=np.float32).view(np.uint32);return ((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)
def f32(v):return (v.astype(np.uint32)<<16).view(np.float32)
def fixture(n,k,m,kind):
 path=fixtures/f'{kind}-{n}-{k}'
 if path.exists():return path
 path.mkdir();rng=np.random.default_rng(n*17+k*31)
 w=bf16(rng.standard_normal((n,k),dtype=np.float32)/np.sqrt(k))
 x=bf16(rng.standard_normal((m,k),dtype=np.float32));b=rng.standard_normal(n,dtype=np.float32)*.01
 if kind=='zero':x[:]=0
 if kind=='cancel':
  # Adjacent equal BF16 weights meet opposite activations, with a small residual.
  w[:,1::2]=w[:,0:k-1:2];x[:,1::2]=bf16(-f32(x[:,0:k-1:2]));x[:,-1]=bf16(np.full(m,2**-10,dtype=np.float32))
 if kind in ['range','wide']:
  exponent=60 if kind=='wide' else 12
  powers=np.linspace(-exponent,exponent,k,dtype=np.float32);w=bf16(f32(w)*np.exp2(powers));x=bf16(f32(x)*np.exp2(-powers));b=np.linspace(-.125,.125,n,dtype=np.float32)
 if kind=='bias-round':
  w[:]=0;x[:]=0;w[:,0]=bf16(np.ones(n,dtype=np.float32));x[:,0]=bf16(np.ones(m,dtype=np.float32));w[:,1]=bf16(np.full(n,2**-8,dtype=np.float32));x[:,1]=bf16(np.ones(m,dtype=np.float32));b[:]=2**-10
 w.tofile(path/'weight.bin');x.tofile(path/'activation.bin');b.tofile(path/'bias.bin')
 (path/'fixture.json').write_text(json.dumps({'N':n,'K':k,'M':m,'kind':kind,'seed':n*17+k*31,'dtype':'bf16 operands; fp32 bias'})+'\n')
 return path
cases=[]
def add(m,n,k,backend='mme',dtype='bf16',bias=0,split=0,kind='random'):cases.append(dict(m=m,n=n,k=k,backend=backend,dtype=dtype,bias=bias,split=split,kind=kind))
if phase=='hybrid':
 for n,k in [(3392,6144),(6144,2048)]:
  for backend in ['mme','hybrid','hybrid4']:
   for bias in [0,1]:add(513,n,k,backend=backend,bias=bias,split=512)
elif phase=='rowdot4':
 for n,k in [(129,259),(3392,6144),(6144,2048),(8192,8192)]:
  for backend in ['rowdot','rowdot4','mme']:
   add(1,n,k,backend=backend,bias=1)
 for kind in ['zero','cancel','wide','bias-round']:
  for dtype in ['bf16','f32']:add(2,135,259,backend='rowdot4',dtype=dtype,bias=1,kind=kind)
elif phase=='launch-ids':
 for backend in ['mme','rowdot']:
  for bias in [0,1]:
   for ids in [False,True]:
    add(1,128,128,backend=backend,bias=bias);cases[-1]['launch_ids']=ids
 for ids in [False,True]:
  add(1,3392,6144);cases[-1]['launch_ids']=ids
elif phase=='overhead-smoke':
 for backend in ['mme','mme-kn','rowdot']:
  for dtype in ['bf16','f32']:
   add(1,129,259,backend=backend,dtype=dtype,bias=1)
 for kind in ['cancel','bias-round']:
  add(2,135,259,backend='rowdot',dtype='f32',bias=1,kind=kind)
elif phase=='overhead':
 for n,k in [(128,128),(129,259),(3392,6144),(6144,2048)]:
  for m in ([1,2] if n<1000 else [1]):
   for backend in ['mme','mme-kn','rowdot']:
    for bias in [0,1]:add(m,n,k,backend=backend,bias=bias)
 for m in [256,257,512,513]:
  for backend in ['mme','mme-kn']:add(m,3392,6144,backend=backend)
 for n,k in [(3392,6144),(6144,2048)]:
  for backend in ['mme','mme-kn']:add(513,n,k,backend=backend,split=512)
 for kind in ['zero','cancel','range','wide','bias-round']:
  for dtype in ['bf16','f32']:add(2,135,259,backend='rowdot',dtype=dtype,bias=1,kind=kind)
 for backend in ['mme','mme-kn','rowdot']:
  for bias in [0,1]:
   for depth in [1,2,4,8,16]:
    add(1,128,128,backend=backend,bias=bias);cases[-1]['depth']=depth
elif phase=='sram':
 for m in [1,64,513]:add(m,3392,6144,bias=1)
 add(513,3392,6144,bias=1,split=512)
 for m in [1,2]:add(m,3392,6144,backend='tpc-native',bias=1)
 add(2,135,259,backend='tpc-native',dtype='f32',bias=1,kind='cancel')
 for backend in ['mme','tpc-native']:
  add(2,135,259,backend=backend,bias=1,kind='bias-round')
  add(2,135,259,backend=backend,bias=0,kind='zero')
  for dtype in ['bf16','f32']:add(2,135,259,backend=backend,dtype=dtype,bias=1,kind='wide')
 add(1,8192,8192,backend='tpc-native')
elif phase=='native':
 for n,k in [(129,259),(3392,6144),(6144,2048)]:
  for backend in ['mme','tpc-native']:
   for m in [1,2]:add(m,n,k,backend=backend,bias=1)
 for m in [64,513]:add(m,3392,6144,bias=1)
 for kind in ['zero','cancel','range','bias-round']:
  for dtype in ['bf16','f32']:add(2,135,259,backend='tpc-native',dtype=dtype,bias=1,kind=kind)
 for backend in ['mme','tpc-native']:add(1,8192,8192,backend=backend)
elif phase=='smoke':
 for backend in ['mme','tpc']:
  for dtype in ['bf16','f32']:
   for m in [1,2]:add(m,129,259,backend,dtype,1)
else:
 grid=[1,2,8,16,64,128,256,257,512,513]
 for m in grid:add(m,3392,6144)
 for m in [1,64,257,513]:add(m,3392,6144,dtype='f32')
 for m in [1,64,513]:add(m,3392,6144,bias=1)
 for m,s in [(257,256),(513,512)]:
  add(m,3392,6144,split=s);add(m,3392,6144,dtype='f32',split=s)
 add(513,3392,6144,bias=1,split=512)
 for m in [1,16,128,513]:add(m,6144,2048)
 add(513,6144,2048,bias=1,dtype='f32');add(513,6144,2048,split=512)
 for m in grid:add(m,129,259,bias=1)
 for kind in ['zero','cancel','range','bias-round']:
  for backend in ['mme','tpc']:
   for dtype in ['bf16','f32']:add(2,135,259,backend,dtype,1,kind=kind)
 for n,k in [(3392,6144),(6144,2048),(129,259)]:
  for m in [1,2]:add(m,n,k,backend='tpc',bias=1)
 for backend in ['mme','tpc']:add(1,8192,8192,backend=backend)
if 'GK_BF16_CASES' in os.environ:cases=json.loads(os.environ['GK_BF16_CASES'])
# Cache CPU references by shape, M, bias and input pattern; every output checked.
references={}
for index,c in enumerate(cases):
 m,n,k=c['m'],c['n'],c['k'];label=f"{index:02d}-{c['backend']}-M{m}-N{n}-K{k}-{c['dtype']}-b{c['bias']}-s{c['split']}-{c['kind']}"
 directory=out/label;directory.mkdir();graphs=directory/'graphs';graphs.mkdir()
 maxm=max(v['m'] for v in cases if (v['n'],v['k'],v['kind'])==(n,k,c['kind']))
 data=fixture(n,k,maxm,c['kind']);env=dict(os.environ,GRAPH_VISUALIZATION='1',GRAPH_VISUALIZATION_DIR=str(graphs),ENABLE_EXPERIMENTAL_FLAGS='true',SRAM_SLICER_GRAPH_VISUALIZATION='1',HABANA_LOGS=str(directory/'logs'))
 env['GK_BF16_CHAIN_DEPTH']=str(c.get('depth',1))
 if c.get('launch_ids'):env['GK_BF16_LAUNCH_IDS']='1'
 else:env.pop('GK_BF16_LAUNCH_IDS',None)
 command=[str(binary),c['backend'],str(m),str(n),str(k),c['dtype'],str(c['bias']),str(c['split']),str(data),str(directory)]
 start=time.monotonic()
 with (directory/'run.log').open('w') as log:
  try:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=100);code=r.returncode
  except subprocess.TimeoutExpired:code=-999
 rec=dict(c,label=label,returncode=code,elapsed_s=time.monotonic()-start,sram_requested='GK_BF16_SRAM' in os.environ)
 lines=(directory/'run.log').read_text(errors='replace').splitlines();events=[];wall=[]
 for line in lines:
  try:v=json.loads(line)
  except ValueError:continue
  if v.get('stage')=='timing':events.append(v['event_us']);wall.append(v['wall_us'])
  if v.get('stage')=='layout':rec['layout']=v
 if code==0:
  key=(m,n,k,c['kind'],c['bias'],c.get('depth',1))
  if key not in references:
   w=f32(np.fromfile(data/'weight.bin',np.uint16)).reshape(n,k).astype(np.float64)
   x=f32(np.fromfile(data/'activation.bin',np.uint16,count=m*k)).reshape(m,k).astype(np.float64)
   b=np.fromfile(data/'bias.bin',np.float32).astype(np.float64) if c['bias'] else np.zeros(n)
   for layer in range(c.get('depth',1)):
    ref=x@w.T+b;scale=np.abs(x)@np.abs(w).T+np.abs(b)
    if layer+1<c.get('depth',1):x=f32(bf16(ref)).astype(np.float64)
   references[key]=(ref,scale);ref.tofile(directory/'reference-fp64.bin')
  ref,scale=references[key];actual=np.fromfile(directory/'output.bin',np.uint16 if c['dtype']=='bf16' else np.float32)
  if c['dtype']=='bf16':actual=f32(actual)
  actual=actual.reshape(m,n).astype(np.float64);error=actual-ref
  rounded=f32(bf16(ref)).astype(np.float64) if c['dtype']=='bf16' else ref.astype(np.float32).astype(np.float64)
  l2=float(np.linalg.norm(error)/max(np.linalg.norm(ref),1e-30));maxabs=float(np.max(np.abs(error)));scaled=float(np.max(np.abs(error)/np.maximum(scale,1e-30)))
  passed=bool(np.isfinite(actual).all() and scaled <= (0.0041 if c['dtype']=='bf16' else 5e-6))
  if c['kind']=='zero' and c['bias']==0:passed=passed and bool(np.all(actual==0))
  rec['correctness']={'checked':int(actual.size),'all_finite':bool(np.isfinite(actual).all()),'relative_l2':l2,'max_abs':maxabs,'rms':float(np.sqrt(np.mean(error**2))),'max_error_over_sum_abs_mac_plus_abs_bias':scaled,'ideal_once_rounded_relative_l2':float(np.linalg.norm(rounded-ref)/max(np.linalg.norm(ref),1e-30)),'different_from_ideal_once_round':int(np.count_nonzero(actual!=rounded)),'screen_pass':passed}
  assert all(np.isfinite(v) and v>0 for v in events+wall) and .4<statistics.median(events)/statistics.median(wall)<1.1
  rec['timing']={'event_us':events,'wall_us':wall,'median_event_us':statistics.median(events),'median_wall_us':statistics.median(wall),'logical_weight_TB_s':n*k*2/statistics.median(events)/1e6,'prepared_weight_TB_s':rec['layout']['weight_bytes']/statistics.median(events)/1e6,'effective_TFLOPS':2*m*n*k*c.get('depth',1)/statistics.median(events)/1e6,'linear_nodes_per_recipe':c.get('depth',1),'median_us_per_linear':statistics.median(events)/c.get('depth',1),'scope':'complete recipe; offline weight transpose, host input copies, compile and cold start excluded'}
 rec['traffic_note']='rowdot reads BF16 weights per token; descriptor coverage is not physical traffic' if c['backend'].startswith('rowdot') else 'prepared BF16 MME/TPC storage; physical transactions unmeasured'
 records.append(rec);print(json.dumps(rec),flush=True);(out/'result.json').write_text(json.dumps({'phase':phase,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'records':records,'deployment_accepted':False},indent=2)+'\n')
failed=[r['label'] for r in records if r['returncode'] or not r.get('correctness',{}).get('screen_pass')]
print(json.dumps({'completed':len(records),'failed':failed}),flush=True)
sys.exit(bool(failed))
