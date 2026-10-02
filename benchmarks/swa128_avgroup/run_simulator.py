"""Actual original/AV-group ELF whole-output and pre-round state bit gates."""
import argparse,hashlib,json,os,re,shutil,subprocess
from pathlib import Path
import numpy as np

def address_proof():
 groups=0
 for pos in range(32768):
  logical0=max(0,pos-127)//128*128;before=[];after=[]
  for logical in (logical0,logical0+128):
   lo=min(128,max(0,pos-127-logical));hi=max(0,min(128,pos-logical+1))
   before.extend(range(logical+lo,logical+hi));t=lo
   while t<hi:
    index=logical+t-(pos-127);count=min(hi-t,16-(index&15))
    assert count>0 and 0<=index<=index+count-1<128 and index//16==(index+count-1)//16
    after.extend(range(logical+t,logical+t+count));t+=count;groups+=1
  assert before==after==list(range(max(0,pos-127),pos+1))
 # Pure byte permutation identity for every score word, including signed zero/NaN payload patterns.
 rng=np.random.default_rng(29160)
 for _ in range(64):
  b=rng.integers(0,256,512,dtype=np.uint8)
  for index in range(128):
   source=b[:256]if index<64 else b[256:];g=(index>>4)&3;local=index&15
   selected=np.tile(source.reshape(4,64)[g],4).reshape(4,64)
   new=np.tile(selected[:,local*4:local*4+4],(1,16)).reshape(-1)
   old=np.tile(source.reshape(4,64)[:,local*4:local*4+4],(1,16)).reshape(4,64)
   old=np.tile(old[g],4)
   assert np.array_equal(new,old)
 return {'all_positions_0_to_32767':True,'group_iterations':groups,'byte_permutation_cases':8192,'missing_or_duplicate_token_reads':0,'scope':'CPU integer/byte proof, separate from actual ELF simulation'}

p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--cases',nargs='+',default=['first','boundary','offset64','allmask','cold1','cold63','cold64','page127','page129','last32766','last32767','cancel32704','sink32767','maskedgroup32767']);a=p.parse_args()
build=a.build.resolve();fixtures=a.fixtures.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
previous=json.loads((fixtures/'result.json').read_text());heads=previous['heads'];slots=previous['slots']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'status':'RUNNING','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'device_verified':False,'heads':heads,'slots':slots,'address_proof':address_proof(),'elf_sha256':{n:sha(build/(n+'.o'))for n in ['quad','group','quad_debug','group_debug']},'records':[]}
for case in a.cases:
 dest=out/case;dest.mkdir();old=fixtures/case
 for name in ['query','key','value','sinks','pages','groups','position']:
  path=old/(name+'.bin');os.link(path,dest/path.name)
 record={'case':case,'input_sha256':{p.name:sha(p)for p in dest.iterdir()},'runs':{}};report['records'].append(record)
 expected=None;raw_state=None
 for name in ['quad','group','quad_debug','group_debug']:
  output=dest/(name+'.bin');command=[str(build/'simulator'),str(heads),str(slots),str(dest),str(output),str(build/(name+'.o'))]
  run={'command':command,'timeout_s':45};record['runs'][name]=run
  try:
   with(dest/(name+'.log')).open('w')as log:r=subprocess.run(command,env=dict(os.environ,TPC_RUNNER='0'),stdout=log,stderr=log,timeout=45)
   run['returncode']=r.returncode;assert r.returncode==0
   bits=np.fromfile(output,dtype=np.uint16);assert len(bits)==heads*128
   if expected is None:expected=bits
   run['BF16_word_mismatches']=int((bits!=expected).sum());assert run['BF16_word_mismatches']==0
   if name.endswith('debug'):
    state={k:np.fromfile(str(output)+'.'+k+'.bin',dtype=np.uint32)for k in ['scores','accum','context']}
    assert all(len(v)==heads*128 for v in state.values())
    if raw_state is None:raw_state=state
    run['FP32_word_mismatches']={k:int((v!=raw_state[k]).sum())for k,v in state.items()};assert not any(run['FP32_word_mismatches'].values())
   run['simulator']=json.loads(re.findall(r'\{"cycles".*\}',(dest/(name+'.log')).read_text())[-1]);run['status']='PASS'
  except Exception as e:run.update(status='FAIL',error=repr(e))
  (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'case':case,'variant':name,**run}),flush=True)
  if run['status']!='PASS':break
 if any(x['status']!='PASS'for x in record['runs'].values()):break
report['status']='PASS_BITS'if len(report['records'])==len(a.cases)and all(len(c['runs'])==4 and all(x['status']=='PASS'for x in c['runs'].values())for c in report['records'])else'FAIL'
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['status']=='PASS_BITS'else 1)
