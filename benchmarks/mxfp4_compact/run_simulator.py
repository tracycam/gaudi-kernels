"""Minimal actual-ELF ISA simulation, explicit TPC_RUNNER=0 and timeout45s."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.mxfp4_compact import prepare
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path)
p.add_argument('--library-dir',required=True,type=Path)
p.add_argument('--case',choices=['native512k32','rows3k31','rows3k32'],required=True);a=p.parse_args()
out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);libdir=a.library_dir.resolve()
kind='native' if a.case.startswith('native') else 'rows';n=512 if kind=='native' else 3
k=31 if a.case=='rows3k31' else 32
rng=np.random.default_rng(27092704)
rows=rng.integers(0,256,(n,(k+1)//2),dtype=np.uint8)
scales=rng.integers(2,253,(n,(k+31)//32),dtype=np.uint8)
scales[:min(n,7),0]=np.array([2,252,127,3,251,126,128],np.uint8)[:min(n,7)]
# Include both signed-zero codes and every nibble value at every chosen scale.
rows[:,:8]=np.arange(8,dtype=np.uint8)*2|((np.arange(8,dtype=np.uint8)*2+1)<<4)
prepared=prepare(rows,scales,logical_k=k)
base=np.array([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],np.float64)
codes=np.empty((n,((k+1)//2)*2),np.uint8);codes[:,::2]=rows&15;codes[:,1::2]=rows>>4
expected_float=np.ldexp(base[codes[:,:k]],np.repeat(scales.astype(np.int32)-127,32,axis=1)[:,:k])
expected=(expected_float.astype(np.float32).view(np.uint32)>>16).astype(np.uint16)
# Exact BF16 is guaranteed only in this decoder's admitted scale domain.
assert np.array_equal((expected.astype(np.uint32)<<16).view(np.float32).astype(np.float64).view(np.uint64),expected_float.view(np.uint64))
if kind=='native':expected=expected.T.copy()
base_bits=(base.astype(np.float32).view(np.uint32)>>16).astype(np.uint16)
table=np.empty(512,np.uint16);table[::2]=base_bits[np.arange(256)&15];table[1::2]=base_bits[np.arange(256)>>4]
rows.tofile(out/'checkpoint-rows.bin');scales.tofile(out/'checkpoint-scales.bin');prepared.weight.tofile(out/'packed.bin');prepared.scales.tofile(out/'scales.bin')
table.tofile(out/'table.bin');expected.tofile(out/'expected-bf16.bin')
source=Path(__file__).with_name('simulator.cpp')
for f in [source,Path(__file__),root/'python/gaudi_kernels/mxfp4_compact.py']:shutil.copy2(f,out/f.name)
library=libdir/'libgaudi_mxfp4_compact_tpc.so';shutil.copy2(library,out/library.name)
compile_command=['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(source),'-L'+str(libdir),'-lgaudi_mxfp4_compact_tpc','-L/usr/lib/habanatools','-ltpc_tests_core_ext','-Wl,-rpath,'+str(libdir)+':/usr/lib/habanatools','-o',str(out/'simulator')]
run_command=[str(out/'simulator'),kind,str(n),str(k),str(out/'packed.bin'),str(out/'scales.bin'),str(out/'table.bin'),str(out/'actual-bf16.bin')]
meta={'case':a.case,'n':n,'k':k,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'compile_command':compile_command,'run_command':run_command,'TPC_RUNNER':'0','timeout_seconds':45,'device_runtime_verified':False,
 'kernel_library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
 'simulator_library_sha256':hashlib.sha256(Path('/usr/lib/habanatools/libtpc_tests_core_ext.so').read_bytes()).hexdigest()}
try:
 with (out/'compile.log').open('w') as log:subprocess.run(compile_command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
 env=dict(os.environ,TPC_RUNNER='0');env['LD_LIBRARY_PATH']=str(libdir)+':/usr/lib/habanatools'
 with (out/'simulation.log').open('w') as log:subprocess.run(run_command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
 actual=np.fromfile(out/'actual-bf16.bin',np.uint16);flat=expected.ravel()
 meta['output_words']=int(actual.size)
 meta['bf16_bit_mismatches']=int(np.count_nonzero(actual!=flat)) if actual.shape==flat.shape else -1
 if actual.shape==flat.shape:
  bad=np.flatnonzero(actual!=flat)[:32]
  meta['first_mismatches']=[{'offset':int(i),'actual':hex(int(actual[i])),'expected':hex(int(flat[i]))} for i in bad]
 meta['state']='isa_simulation_passed' if meta['bf16_bit_mismatches']==0 else 'numerical_failure'
except (subprocess.SubprocessError,OSError) as error:meta.update(state='failed',error=str(error))
meta['artifacts']={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}
(out/'result.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({k:v for k,v in meta.items() if k!='artifacts'},indent=2))
raise SystemExit(0 if meta['state']=='isa_simulation_passed' else 1)
