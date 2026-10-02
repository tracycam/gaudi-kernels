"""Actual ELF simulator versus independent BF16 arithmetic and layout oracle."""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build=a.build.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2];src=root/'benchmarks/qkv_postprocess/simulator.cpp';binary=out/'simulator'
command=['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(src),'-L'+str(build),'-lgaudi_qkv_postprocess_tpc','-L/usr/lib/habanatools','-ltpc_tests_core_ext','-Wl,-rpath,'+str(build)+':/usr/lib/habanatools','-o',str(binary)]
with (out/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
torch.set_num_threads(2);torch.manual_seed(280935);m=3
x=(torch.randn(m,3392)*8).bfloat16();x[:,::37]=0;x[:,::41]=-torch.tensor(0.,dtype=torch.bfloat16)
phase=torch.randn(m,1,32);c=phase.cos().bfloat16().repeat(1,1,2);s=phase.sin().bfloat16().repeat(1,1,2);c[0].fill_(1);s[0].zero_();c[1].zero_();s[1].fill_(1)
for name,t in [('qkv',x),('cosine',c),('sine',s)]:t.view(torch.int16).numpy().tofile(out/(name+'.bin'))
records=[]
for rmode in ['bf16','f32']:
 for vmode in ['bf16','f32']:
  kind=rmode+'_'+vmode;dest=out/kind;dest.mkdir();query=x[:,:3072].view(m,16,192).clone();key=x[:,3072:3264].view(m,1,192).clone();value=x[:,3264:].view(m,1,128).clone()
  for t in [query,key]:
   a=t[...,:64];partner=torch.cat((-a[...,32:64],a[...,:32]),-1)
   if rmode=='bf16':a.copy_((a*c).bfloat16()+(partner*s).bfloat16())
   else:a.copy_((a.float()*c.float()+partner.float()*s.float()).bfloat16())
  scale=torch.tensor(.612,dtype=torch.bfloat16 if vmode=='bf16' else torch.float32).float()
  value=(value.float()*scale).bfloat16()
  env=dict(os.environ,TPC_RUNNER='0',LD_LIBRARY_PATH=str(build)+':/usr/lib/habanatools')
  command_run=[str(binary),kind,str(m),str(out),str(dest)]
  with (dest/'run.log').open('w') as log:run=subprocess.run(command_run,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=45)
  record={'variant':kind,'command':command_run,'returncode':run.returncode,'mismatches':{},'nonzero_mismatches':{},'zero_sign_mismatches':{}};records.append(record)
  if run.returncode==0:
   for name,t in [('query',query),('key',key),('value',value)]:
    expected=t.contiguous().view(torch.int16).numpy().view(np.uint16);expected.tofile(dest/(name+'-expected.bin'));actual=np.fromfile(dest/(name+'.bin'),np.uint16).reshape(expected.shape)
    different=actual!=expected;both_zero=((actual&0x7fff)==0)&((expected&0x7fff)==0)
    record['mismatches'][name]=int(np.count_nonzero(different));record['zero_sign_mismatches'][name]=int(np.count_nonzero(different&both_zero));record['nonzero_mismatches'][name]=int(np.count_nonzero(different&~both_zero))
  (out/'result.json').write_text(json.dumps({'state':'RUNNING','records':records,'device_verified':False},indent=2)+'\n');assert run.returncode==0 and not any(record['nonzero_mismatches'].values()),record
shutil.copy2(src,out/'source.cpp');shutil.copy2(Path(__file__),out/'driver.py')
meta={'state':'COMPLETED_WITH_ZERO_SIGN_DIFFERENCES','full_bitwise_match':all(not any(r['mismatches'].values()) for r in records),'nonzero_bits_match':True,'vendor_arithmetic_qualified':False,'records':records,'device_verified':False,'build_command':command,'compiler_source_commit':json.loads((build/'build.json').read_text())['source_commit'],'tpc_library_sha256':hashlib.sha256((build/'libgaudi_qkv_postprocess_tpc.so').read_bytes()).hexdigest(),'simulator_library_sha256':hashlib.sha256(Path('/usr/lib/habanatools/libtpc_tests_core_ext.so').read_bytes()).hexdigest()}
meta['files_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='result.json'};(out/'result.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'state':meta['state'],'variants':len(records),'words_per_variant':m*3392}))
