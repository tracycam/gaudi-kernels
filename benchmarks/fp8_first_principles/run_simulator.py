"""Minimal local ISA simulator; TPC_RUNNER is forced to0, process timeout45s."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from cpu_gate import C,FLOOR,bf16_bits,bf16_float,half_code,ocp_code

p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--library-dir',type=Path,required=True);p.add_argument('--table',type=Path,required=True)
p.add_argument('--case',choices=['basic129','clamp129','extreme513','mantissas129'],default='basic129');p.add_argument('--variant',choices=['baseline','amax16','lut_single','lut_single4'],default='lut_single4');a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
source=Path(__file__).with_name('simulator.cpp');shutil.copy2(source,out/'simulator.cpp')
shutil.copy2(Path(__file__),out/'run_simulator.py');shutil.copy2(Path(__file__).with_name('cpu_gate.py'),out/'cpu_gate.py')
shutil.copy2(a.table,out/'table.bin')
k=513 if a.case=='extreme513' else 129
m=128 if a.case=='mantissas129' else 1
rng=np.random.default_rng(129);bits=bf16_bits(rng.uniform(-.3,.3,k))
if a.case=='basic129':
 bits[:6]=bf16_bits([1.6953125,1.1086463928222656e-05,-1.1086463928222656e-05,0.,-0.,-1.6953125])
elif a.case=='clamp129':
 bits=np.resize(np.array([0,0x8000,1,0x8001,127,0x807f,128,0x8080,0x2edb,0xaedb],np.uint16),k)
elif a.case=='extreme513':
 bits=np.resize(np.array([0,0x8000,1,0x8001,127,0x807f,128,0x8080,0x7f7f,0xff7f,0x3f80,0xbf80],np.uint16),k)
else:
 maxima=(1+np.arange(128,dtype=np.float32)/128)[:,None]
 x=bf16_float(bf16_bits(rng.uniform(-.9,.9,(128,k))*maxima))
 x[:,0]=maxima[:,0];x[:,1]=-maxima[:,0];x[:,2]=0.;x[:,3]=-0.
 x[:,4]=bf16_float(bf16_bits(maxima[:,0]*1e-5));x[:,5]=-x[:,4]
 x[89,6]=np.float32(1.1086463928222656e-05)
 bits=bf16_bits(x)
x=bf16_float(bits).reshape(m,k);scale=(np.maximum(np.max(np.abs(x),axis=1,keepdims=True),FLOOR)*C).astype(np.float32)
expected=half_code(ocp_code((x.astype(np.float64)/scale.astype(np.float64)).astype(np.float32))).ravel()
bits.tofile(out/'input.bin');expected.tofile(out/'expected-q.bin');(scale*2).ravel().astype(np.float32).tofile(out/'expected-scale.bin')
libdir=a.library_dir.resolve()
compile_command=['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(source),'-L'+str(libdir),
 '-lgaudi_fp8_first_principles_tpc','-L/usr/lib/habanatools','-ltpc_tests_core_ext',
 '-Wl,-rpath,'+str(libdir)+':/usr/lib/habanatools','-o',str(out/'simulator')]
run_command=[str(out/'simulator'),str(k),str(m),str(out/'input.bin'),str(out/'table.bin'),str(out/'q.bin'),str(out/'scale.bin'),a.variant,str(libdir/'baseline.o')]
meta={'case':a.case,'M':m,'K':k,'variant':a.variant,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'compile_command':compile_command,'run_command':run_command,'TPC_RUNNER':'0','timeout_seconds':45,
 'device_runtime_verified':False,'kernel_library_sha256':hashlib.sha256((libdir/'libgaudi_fp8_first_principles_tpc.so').read_bytes()).hexdigest()}
try:
 with (out/'compile.log').open('w') as log:subprocess.run(compile_command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
 env=dict(os.environ,TPC_RUNNER='0');env['LD_LIBRARY_PATH']=str(libdir)+':/usr/lib/habanatools'
 with (out/'simulation.log').open('w') as log:subprocess.run(run_command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
 actual=np.fromfile(out/'q.bin',np.uint8);actual_scale=np.fromfile(out/'scale.bin',np.uint32)
 meta['output_bytes']=int(actual.size)
 meta['quant_byte_mismatches']=int(np.count_nonzero(actual!=expected)) if actual.shape==expected.shape else -1
 meta['scale_word_matches']=bool(np.array_equal(actual_scale,(scale*2).ravel().astype(np.float32).view(np.uint32)))
 meta['state']='isa_simulation_passed' if meta['quant_byte_mismatches']==0 and meta['scale_word_matches'] else 'numerical_failure'
except (subprocess.SubprocessError,OSError) as error:meta.update(state='failed',error=str(error))
meta['artifacts']={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}
(out/'result.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k!='artifacts'},indent=2))
raise SystemExit(0 if meta['state']=='isa_simulation_passed' else 1)
