"""Offline deployed-ISA audit of natural GP column order; never acquires a device."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
import numpy as np
from elftools.elf.elffile import ELFFile

p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--legacy-elf',type=Path,required=True);p.add_argument('--bundled-literal-elf',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
lib=a.library.resolve();sim=out/'simulator'
cmd=['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(root/'benchmarks/moe_route_tiles/isa_simulator.cpp'),'-L/usr/lib/habanatools','-Wl,-rpath,/usr/lib/habanatools','-ltpc_tests_core_ext','-ldl','-o',str(sim)]
(out/'compile-command.json').write_text(json.dumps(cmd)+'\n')
with (out/'compile.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=45)
def run(name,guid,inputs,shape,params):
 d=out/name;d.mkdir();spec=[guid,str(len(inputs)),'1',str(len(params)),*map(str,params)]
 for i,x in enumerate(inputs):
  x=np.ascontiguousarray(x);path=d/f'input{i}.bin';x.tofile(path);typ={np.dtype('float32'):'f32',np.dtype('int32'):'i32'}[x.dtype]
  spec.extend([typ,str(x.ndim),*map(str,x.shape[::-1]),str(path)])
 path=d/'output.bin';spec.extend(['bf16',str(len(shape)),*map(str,shape[::-1]),str(path)]);(d/'spec.txt').write_text(' '.join(spec)+'\n')
 command=[str(sim),str(lib),str(d/'spec.txt'),'-'];(d/'command.json').write_text(json.dumps(command)+'\n')
 with (d/'stdout.log').open('w') as f:subprocess.run(command,env={**os.environ,'TPC_RUNNER':'0'},stdout=f,stderr=subprocess.STDOUT,check=True,timeout=45)
 return np.fromfile(path,dtype='u2').reshape(shape)
def bf(x):
 u=np.asarray(x,dtype='f4').view('u4');return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('u2')
def f32(x):return (x.astype('u4')<<16).view('f4')
def cpu(p):
 g=f32(bf(p[...,:256]));u=f32(bf(p[...,256:]));s=(g.astype('f8')/(1+np.exp(-g.astype('f8')))).astype('f4')
 return bf(f32(bf(s))*u)
def ulp(a,b):
 # Monotone BF16 integer order across signs, with +/-zero adjacent.
 def ordered(x):return np.where(x&0x8000,0xffff-x,x.astype('i4')+0x8000)
 return np.abs(ordered(a).astype('i4')-ordered(b).astype('i4'))
rng=np.random.default_rng(20260928);cases={}
cases['random']=rng.uniform(-8,8,(2,3,512)).astype('f4')
# At g=16 the literal SFU and mathematical SiLU round to BF16 16.
# Unique, nonperiodic up values make every K permutation observable without SFU tolerance.
tag=np.empty((2,3,512),dtype='f4');tag[...,:256]=16
for row in tag.reshape(-1,512):row[256:]=f32(rng.permutation(np.arange(0x3d00,0x3e00,dtype='u2')))
cases['positive16_coordinate_tags']=tag
cases['random_bf16']=f32(bf(rng.uniform(-4,4,(2,3,512)).astype('f4')))
records=[]
for name,partial in cases.items():
 b,c,_=partial.shape;rows=b*c;valid=np.full(b,c,dtype='i4');status=np.zeros(1,dtype='i4')
 got=run(name+'-natural','gk_route_tile_gate_bf16_v1',[partial,valid,status],(b,c,256),(0,))
 # Native GEMV returns evens then odds within each group of 128 natural columns.
 n=np.arange(512);physical_index=(n//128)*128+(n%2)*64+(n%128)//2
 legacy=np.empty_like(partial.reshape(rows,512));legacy[:,physical_index]=partial.reshape(rows,512)
 old=run(name+'-native','gk_mxfp4_graph_literal_gate',[legacy.reshape(1,-1),np.zeros((rows,1),dtype='i4')],(rows,256),(1,)).reshape(got.shape)
 split=np.zeros((rows,3,512),dtype='f4');split[:,0]=legacy
 old3=run(name+'-native3','gk_mxfp4_graph_literal_gate',[split.reshape(1,-1),np.zeros((rows,1),dtype='i4')],(rows,256),(3,)).reshape(got.shape)
 want=cpu(partial);delta=ulp(got,want);mismatch=np.argwhere(got!=want)
 record=dict(case=name,words=got.size,natural_vs_native_bit_mismatches=int(np.count_nonzero(got!=old)),natural_vs_native3_bit_mismatches=int(np.count_nonzero(got!=old3)),natural_vs_cpu_bit_mismatches=int(mismatch.shape[0]),natural_vs_cpu_max_bf16_ulp=int(delta.max()),witnesses=[])
 for index in mismatch[:8]:
  i=tuple(index);record['witnesses'].append(dict(index=index.tolist(),gate=float(partial[i[:-1]+(i[-1],)]),up=float(partial[i[:-1]+(i[-1]+256,)]),got_bits=int(got[i]),cpu_bits=int(want[i])))
 if name=='positive16_coordinate_tags':
  n=np.arange(256);permuted=got[...,((n//128)*128+(n%2)*64+(n%128)//2)]
  record['injected_native_order_output_vs_cpu_mismatches']=int(np.count_nonzero(permuted!=want))
  assert record['injected_native_order_output_vs_cpu_mismatches']>1400
 np.save(out/(name+'-cpu.npy'),want);records.append(record)
elfs=[]
for path in (a.legacy_elf.resolve(),a.bundled_literal_elf.resolve()):
 with path.open('rb')as f:text=ELFFile(f).get_section_by_name('.text').data()
 elfs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),text_sha256=hashlib.sha256(text).hexdigest()))
same_text=elfs[0]['text_sha256']==elfs[1]['text_sha256']
passed=same_text and all(x['natural_vs_native_bit_mismatches']==0 and x['natural_vs_native3_bit_mismatches']==0 and (x['natural_vs_cpu_bit_mismatches']==0 if x['case']=='positive16_coordinate_tags' else True) for x in records)
report=dict(status='PASS_LANE_ORDER_ONLY' if passed else 'FAIL',records=records,device_tested=False,scope='Natural GP column order and elementwise gate only. Random mathematical SiLU differences are diagnostics, NOT an accepted numerical tolerance. Coordinate-tag case and old-literal comparisons require exact bits; neither establishes compiled MME or model correctness.',legacy_vs_bundled_text_identical=same_text,elfs=elfs,library=str(lib),library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),simulator_sha256=hashlib.sha256(sim.read_bytes()).hexdigest())
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if not passed:raise SystemExit(1)
