"""Actual vendor-ELF versus scalar/vector fusion, with separate FP64 truth."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--vendor',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--all-ids',action='store_true');a=p.parse_args()
build=a.build.resolve();vendor=a.vendor.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
result={'status':'STARTED','TPC_RUNNER':0,'device_qualified':False,'cases':[],
 'acceptance':'IDs exact; scalar/vector bits exact; actual denominator bits exact vendor; finite weights <=1 ULP vendor; exceptional class/sign exact; FP64 truth separate',
 'elf_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [build/'scalar.o',build/'vector.o',*[vendor/(n+'.o') for n in ('gather','reduce','divide','multiply')]]}}
def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
def run(cmd,where,log):
 with (where/log).open('w') as f:f.write(json.dumps(cmd)+'\n');f.flush();subprocess.run(cmd,cwd=where,env=dict(os.environ,TPC_RUNNER='0'),stdout=f,stderr=subprocess.STDOUT,timeout=45,check=True)
def primitive(mode,first,second,output,where):
 params=json.loads((vendor/(mode+'.json')).read_text())['scalar_params']
 run([str(vendor/'simulator'),str(vendor/(mode+'.o')),mode,first,second,output,str(len(params)),*[str(v) for v in params]],where,mode+'.log')
def same(x,y):return bool(np.all((x.view('u4')==y.view('u4'))|(np.isnan(x)&np.isnan(y))))
def ulp(x,y):
 finite=np.isfinite(x)&np.isfinite(y)
 if not finite.any():return 0
 def ordered(v):
  b=v.view('u4').astype(np.int64);return np.where(b&0x80000000,0x80000000-(b&0x7fffffff),0x80000000+b)
 return int(np.max(np.abs(ordered(x[finite])-ordered(y[finite]))))
rng=np.random.default_rng(928384);ids=np.array([383,0,255,128,63,64,129,382],np.int32)
cases=[('normal',rng.uniform(.001,1,384).astype('f4'),2.5,True),
 ('ties',np.full(384,.5,'f4'),2.5,True),
 ('near_ulp',np.resize(np.array([.5,np.nextafter(np.float32(.5),np.float32(1)),np.nextafter(np.float32(.5),np.float32(0))],'f4'),384),1.,True),
 ('no_renorm',rng.uniform(.001,1,384).astype('f4'),2.5,False),
 ('small_normal',rng.uniform(1e-20,1e-19,384).astype('f4'),1.,True),
 ('zero',np.zeros(384,'f4'),2.5,True),('subnormal',np.full(384,1e-40,'f4'),1.,True),
 ('subnormal_no_renorm',np.full(384,1e-40,'f4'),1.,False),
 ('factor_rounds_to_one_but_guard_true',np.full(384,1e-40,'f4'),1.+2**-30,False),
 ('negative_zero_no_scale',np.full(384,-0.,'f4'),1.,False)]
mixed=np.full(384,1e-20,'f4');mixed[ids[0]]=1e-40;cases.append(('mixed_subnormal',mixed,1.,True))
for i in range(16):
 if i<8:s=rng.uniform(0,1,384).astype('f4')
 else:s=((rng.integers(1,128,384,dtype='u4')<<23)|rng.integers(0,1<<23,384,dtype='u4')).view('f4');s=np.minimum(s,1.)
 cases.append((f'random-{i}',s,[1.,2.5][i%2],True))
try:
 for name,scores,factor,renorm in cases:
  d=out/name;d.mkdir();scores.tofile(d/'scores.bin');ids.tofile(d/'ids.bin');np.array([factor],'f4').tofile(d/'factor.bin')
  primitive('gather','scores.bin','ids.bin','vendor-gather.bin',d)
  primitive('reduce','vendor-gather.bin','ids.bin','vendor-sum.bin',d)
  if renorm:primitive('divide','vendor-gather.bin','vendor-sum.bin','vendor-quotient.bin',d);current='vendor-quotient.bin'
  else:current='vendor-gather.bin'
  if factor!=1.:primitive('multiply',current,'factor.bin','vendor-weight.bin',d)
  else:(d/'vendor-weight.bin').write_bytes((d/current).read_bytes())
  results={}
  for mode in ('scalar','vector'):
   run([str(build/'simulator'),str(build/(mode+'.o')),'scores.bin','ids.bin',mode+'-weights.bin',mode+'-ids.bin',mode+'-sum.bin',str(factor),str(int(renorm))],d,mode+'.log')
   results[mode]=np.fromfile(d/(mode+'-weights.bin'),'f4')
   assert (d/(mode+'-ids.bin')).read_bytes()==(d/'ids.bin').read_bytes()
   assert (d/(mode+'-sum.bin')).read_bytes()==(d/'vendor-sum.bin').read_bytes(),(name,mode,'sum')
  expected=np.fromfile(d/'vendor-weight.bin','f4');actual=results['vector']
  finite=np.isfinite(expected);assert np.array_equal(np.isfinite(actual),finite),(name,'finite class')
  assert np.array_equal(np.isnan(actual),np.isnan(expected)) and np.array_equal(np.isinf(actual),np.isinf(expected))
  assert np.array_equal(np.signbit(actual[~np.isnan(expected)]),np.signbit(expected[~np.isnan(expected)])),(name,'non-NaN sign')
  assert same(results['scalar'],actual),(name,'fetch routes differ')
  error=ulp(actual,expected);assert error<=1,(name,'vendor ULP',error)
  with np.errstate(all='ignore'):
   truth=scores[ids].astype('f8');truth=truth/truth.sum() if renorm else truth;truth=truth*factor
  truth.tofile(d/'fp64-truth.bin')
  finite_truth=np.isfinite(truth)
  row={'name':name,'factor':factor,'renormalize':renorm,'ordered_ids_exact':True,'denominator_bits_exact_vendor':True,
       'scalar_vector_bits_equal':True,'weights_vendor_bits_equal':same(actual,expected),'max_vendor_ulp':error,
       'finite_truth':bool(finite_truth.all()),'finite_vendor':bool(finite.all()),
       'max_abs_vs_fp64':float(np.max(np.abs(actual.astype('f8')-truth))) if finite.all() and finite_truth.all() else None}
  result['cases'].append(row);save()
 if a.all_ids:
  d=out/'all-ids';d.mkdir();scores=(np.arange(384,dtype='f4')+1)/1024;scores.view('u4')[:6]=[0,0x80000000,1,0x007fffff,0x00800000,0x7fc01234];scores.tofile(d/'scores.bin')
  for offset in range(384):
   c=d/str(offset);c.mkdir();ii=np.array([(offset+73*slot)%384 for slot in range(8)],'i4');ii.tofile(c/'ids.bin')
   run([str(build/'simulator'),str(build/'vector.o'),'../scores.bin','ids.bin','weights.bin','out-ids.bin','sum.bin','1','0'],c,'run.log')
   assert (c/'ids.bin').read_bytes()==(c/'out-ids.bin').read_bytes();assert np.array_equal(np.fromfile(c/'weights.bin','u4'),scores[ii].view('u4'))
  result['register_map']={'all384_ids_in_all8_slots_exact':True,'checked_words':3072,'special_raw_bits_preserved_without_math':True}
 result['status']='VENDOR_ELF_ULP_GATE_PASS_NOT_DEVICE_QUALIFIED'
except Exception as e:result.update(status='FAIL',error=repr(e));raise
finally:save()
print(json.dumps({'status':result['status'],'cases':len(result['cases']),'max_vendor_ulp':max((v['max_vendor_ulp'] for v in result['cases']),default=None)}))
