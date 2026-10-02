"""CPU-only actual deployed ELF/vendor ELF gate; no full-attention execution."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--tpc-library',type=Path,required=True);p.add_argument('--vendor-elf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
commands=[]
for name,source in [('candidate','benchmarks/qkv_post_full/cache_simulator.cpp'),('vendor','benchmarks/qkv_postprocess/vendor_simulator.cpp')]:
 cmd=['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(root/source),'-L/usr/lib/habanatools','-ltpc_tests_core_ext','-Wl,-rpath,/usr/lib/habanatools','-o',str(out/name)]
 if name=='candidate':cmd.extend([str(a.tpc_library.resolve()),'-Wl,-rpath,'+str(a.tpc_library.resolve().parent)])
 commands.append(cmd)
 with (out/(name+'-build.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
torch.set_num_threads(2);torch.manual_seed(290044);records=[]
for theta in (10000,1000000):
 folder=out/str(theta);folder.mkdir();(folder/'candidate').mkdir();(folder/'vendor').mkdir()
 x=(torch.randn(3,3392)*4).bfloat16();x[:,::41]=0;x[:,::43]=-torch.tensor(0.,dtype=torch.bfloat16)
 positions=torch.tensor([0,128,32767]);freq=(float(theta)**(-torch.arange(0,64,2,dtype=torch.float64)/64));phase=positions.double()[:,None]*freq
 c=phase.cos().bfloat16().repeat(1,2);s=phase.sin().bfloat16().repeat(1,2)
 for name,t in [('qkv',x),('cosine',c),('sine',s)]:t.view(torch.int16).numpy().tofile(folder/(name+'.bin'))
 env=dict(os.environ,TPC_RUNNER='0')
 for name,cmd in [('candidate',[str(out/'candidate'),'cache','3',str(folder),str(folder/'candidate')]),('vendor',[str(out/'vendor'),str(a.vendor_elf.resolve()),str(folder),str(folder/'vendor')])]:
  commands.append(cmd)
  with (folder/(name+'.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=45,check=True)
 expected=[]
 for t in (x[:,:3072].reshape(3,16,192).clone(),x[:,3072:3264].reshape(3,1,192).clone()):
  first=t[...,:64];partner=torch.cat((-first[...,32:],first[...,:32]),-1)
  t[...,:64]=((first*c[:,None,:]).bfloat16()+(partner*s[:,None,:]).bfloat16()).bfloat16();expected.append(t)
 expected.append((x[:,3264:].float()*torch.tensor(.612,dtype=torch.bfloat16).float()).bfloat16())
 row={'synthetic_cache_theta':theta,'positions':[0,128,32767],'mismatches':{},'cpu_zero_sign_mismatches':{},'cpu_nonzero_mismatches':{},'scope':'same explicit synthetic cache payload in candidate/vendor/oracle; does not assert model cache construction theta'}
 for name,want in zip(('query','key','value'),expected):
  raw=want.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1);actual=np.fromfile(folder/'candidate'/(name+'.bin'),np.uint16)
  raw.tofile(folder/(name+'-cpu.bin'));row['mismatches'][name+'_cpu']=int(np.count_nonzero(actual!=raw))
  zeros=((actual&0x7fff)==0)&((raw&0x7fff)==0)
  row['cpu_zero_sign_mismatches'][name]=int(np.count_nonzero((actual!=raw)&zeros))
  row['cpu_nonzero_mismatches'][name]=int(np.count_nonzero((actual!=raw)&~zeros))
  if name!='value':row['mismatches'][name+'_vendor']=int(np.count_nonzero(actual!=np.fromfile(folder/'vendor'/(name+'.bin'),np.uint16)))
 records.append(row);(out/'partial-result.json').write_text(json.dumps(records,indent=2)+'\n');assert not any(row['cpu_nonzero_mismatches'].values()) and all(not value for key,value in row['mismatches'].items() if key.endswith('_vendor')),row
result={'status':'PASS_VENDOR_QK_BITS_AND_CPU_NONZERO_BITS','cpu_full_bitwise_match':all(not value for r in records for key,value in r['mismatches'].items() if key.endswith('_cpu')),'device_accessed':False,'full_attention_device_qualified':False,'records':records,'commands':commands,'tpc_library_sha256':hashlib.sha256(a.tpc_library.read_bytes()).hexdigest(),'vendor_elf_sha256':hashlib.sha256(a.vendor_elf.read_bytes()).hexdigest(),'scope':'Q/K actual vendor ELF bitwise match; V uses previous device qualification plus CPU nonzero-bits gate here. CPU signed-zero disagreement retained; full consumer capture remains untested'}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'checked_candidate_output_words':2*3*3392}))
