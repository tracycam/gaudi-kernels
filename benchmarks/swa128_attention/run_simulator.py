"""Bounded ISA-only minimal cases, actual compiled ELF; never acquires a card."""
import argparse,hashlib,json,os,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from oracle import reference,requested_bytes,full_reference
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['bf16','fp32','fp32_fast'],default='bf16');p.add_argument('--flat',action='store_true');p.add_argument('--heads',type=int,default=2);a=p.parse_args();root=Path(__file__).resolve().parents[2];build=a.build.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);binary=out/'simulator'
command=['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(root/'benchmarks/swa128_attention/simulator.cpp'),'-L'+str(build),'-lgaudi_swa128_attention_tpc','-L/usr/lib/habanatools','-ltpc_tests_core_ext','-Wl,-rpath,'+str(build)+':/usr/lib/habanatools','-o',str(binary)]
with (out/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
torch.set_num_threads(2);torch.manual_seed(281724);heads=a.heads;q=torch.randn(1,heads,192).bfloat16();k=torch.randn(512,1,192).bfloat16();v=torch.randn(512,1,128).bfloat16();records=[]
for name,pages,starts,pos,sink in [('first',[-1,1],[-128,0],0,torch.full((heads,),-torch.inf)),('boundary',[3,1],[0,128],128,torch.tensor([-torch.inf,.5]).repeat((heads+1)//2)[:heads]),('offset64',[3,1],[128,256],320,torch.tensor([0.,1.]).repeat((heads+1)//2)[:heads]),('allmask',[-1,-1],[-128,0],0,torch.tensor([-torch.inf,1.]).repeat((heads+1)//2)[:heads])]:
 dest=out/name;dest.mkdir();s=sink.bfloat16();expected,stages=reference(q,k,v,pages,starts,pos,s)
 if a.mode.startswith('fp32'):
  high,expected=full_reference(q,k,v,pages,starts,pos,s);stages={'complete_fp64':high,'final_bf16':expected}
 for field,t in [('query',q),('key',k),('value',v),('sinks',s),('expected',expected)]:t.contiguous().view(torch.int16).numpy().tofile(dest/(field+'.bin'))
 for field,values in [('pages',pages),('starts',starts),('position',[pos])]:np.array(values,dtype=np.int32).tofile(dest/(field+'.bin'))
 torch.save(stages,dest/'oracle-stages.pt');cmd=[str(binary),str(heads),str(dest),str(dest),a.mode,'flat' if a.flat else 'heads'];env=dict(os.environ,TPC_RUNNER='0',LD_LIBRARY_PATH=str(build)+':/usr/lib/habanatools')
 rec={'case':name,'command':cmd,'timeout_s':45,'requested_bytes':requested_bytes(pages,starts,pos,512,heads)};records.append(rec)
 try:
  with (dest/'run.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=45)
  rec['returncode']=r.returncode
  if not r.returncode:
   bits=np.fromfile(dest/'output.bin',dtype=np.uint16);actual=torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).reshape(expected.shape);delta=actual.double()-expected.double();rec.update(finite=bool(torch.isfinite(actual).all()),bit_mismatches=int((actual.view(torch.int16)!=expected.view(torch.int16)).sum()),relative_l2=float(delta.norm()/expected.double().norm().clamp_min(1e-30)),max_abs=float(delta.abs().max()));rec['status']='PASS_TOLERANCE_ONLY' if rec['finite'] and rec['relative_l2']<.006 else 'FAIL_NUMERICS'
  else:rec['status']='FAIL_SIMULATOR'
 except subprocess.TimeoutExpired:rec['status']='TIMEOUT'
 (out/'result.json').write_text(json.dumps({'status':'RUNNING','records':records,'device_verified':False},indent=2)+'\n');print(json.dumps(rec),flush=True)
 if not rec['status'].startswith('PASS'):break
for name in ['benchmarks/swa128_attention/simulator.cpp','benchmarks/swa128_attention/run_simulator.py','benchmarks/swa128_attention/oracle.py']:
 dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dst)
result={'status':'PASS_MINIMAL_ISA_TOLERANCE' if len(records)==4 and all(r['status'].startswith('PASS') for r in records) else 'INCOMPLETE_OR_FAILED','records':records,'mode':a.mode,'query_layout':'flat' if a.flat else 'heads','device_verified':False,'vendor_hpu_numerics_verified':False,'build_command':command,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'elf_sha256':hashlib.sha256((build/({'bf16':'head.o','fp32':'head_fp32.o','fp32_fast':'head_fast.o'}[a.mode])).read_bytes()).hexdigest(),'simulator_sha256':hashlib.sha256(Path('/usr/lib/habanatools/libtpc_tests_core_ext.so').read_bytes()).hexdigest()};result['files_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='result.json'};(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
