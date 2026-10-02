"""Compare actual old/new decoder ELFs; classify common FP32/zero boundaries."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np
from elftools.elf.elffile import ELFFile

p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
commands=[];identities={}
def digest(x):return hashlib.sha256(x).hexdigest()
for name,path in [('old',a.old),('new',a.new)]:
 raw=path.read_bytes();elf=ELFFile(io.BytesIO(raw));syms={s.name:s for s in elf.get_section_by_name('.symtab').iter_symbols()};first=syms['_binary_block_decode_o_start'];last=syms['_binary_block_decode_o_end'];section=elf.get_section(first['st_shndx']);offset=section['sh_offset']+first['st_value']-section['sh_addr'];blob=raw[offset:offset+last['st_value']-first['st_value']]
 target=out/(name+'.elf');target.write_bytes(blob);text=ELFFile(io.BytesIO(blob)).get_section_by_name('.text').data();(out/(name+'.text')).write_bytes(text)
 cmd=['/usr/bin/tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',str(target)];commands.append(cmd);dis=subprocess.check_output(cmd,text=True);(out/(name+'.dis')).write_text(dis)
 ops=Counter()
 for line in dis.splitlines():
  m=re.match(r'\s*[0-9a-f]+:\s+(.*)',line)
  if m:
   for slot in m[1].split('//')[0].split(';'):
    if slot.strip() and slot.strip()!='nop':ops[slot.strip().split()[0]]+=1
 identities[name]=dict(path=str(path.resolve()),library_sha256=digest(raw),elf_sha256=digest(blob),text_sha256=digest(text),static_opcodes=dict(ops))
assert identities['new']['library_sha256']=='121173a4dca0c293db042f904cdb16d2d74ab58022a9cb2f76d9a4785b776740'
cmd=['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(root/'benchmarks/fp8_native_lanes_review/simulator.cpp'),'-L/usr/lib/habanatools','-Wl,-rpath,/usr/lib/habanatools','-ltpc_tests_core_ext','-ldl','-o',str(out/'simulator')];commands.append(cmd)
with(out/'build.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=45)
scale_bits=np.array([0x3f800000,0x3f800001,0x3f808000,0x3f7fffff,0x3eaaaaab,0xbf800000,0,0x80000000,0x00800000,0x007fffff,0x00000001,0x00010000,0x01000000,0x7f7fffff,0x7f000000,0x40000000,0x3f000000,0x807fffff,0x7f800000,0x7fc01234],dtype=np.uint32)
records=[]
for name,n,finite in [('finite_codes_tail',129,True),('all_codes_diagnostic',1,False)]:
 folder=out/name;folder.mkdir();g=len(scale_bits)
 codes=np.array(list(range(120))+list(range(128,248)) if finite else list(range(256)),dtype=np.uint8)
 index=np.arange(g*n*128).reshape(g,n,128);w=codes[(index*17+index//128)%len(codes)]
 assert np.array_equal(np.unique(w),codes)
 # Each block uses a distinct scale permutation, including the N129 tail.
 scales=np.stack([np.roll(scale_bits,3*i)for i in range((n+127)//128)]).view(np.float32)
 w.tofile(folder/'weights.bin');scales.tofile(folder/'scales.bin')
 results={}
 for variant,library in [('old',a.old),('new',a.new)]:
  cmd=[str(out/'simulator'),str(library.resolve()),str(n),str(g),str(folder/'weights.bin'),str(folder/'scales.bin'),str(folder/(variant+'.bin'))];commands.append(cmd)
  with(folder/(variant+'.log')).open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,TPC_RUNNER='0'),check=True,timeout=45)
  results[variant]=np.fromfile(folder/(variant+'.bin'),np.uint16).reshape(n,g*128)
 mismatches=int(np.count_nonzero(results['old']!=results['new']))
 record=dict(case=name,N=n,groups=g,words_checked=n*g*128,old_new_bit_mismatches=mismatches,finite_native_codes=finite)
 if finite:
  code=w.astype(np.int32);exponent=(code>>3)&15;mantissa=code&7
  values=np.where(exponent==0,mantissa*np.float64(2**-9),(1+mantissa/8)*np.exp2(exponent-7))*np.where(code&128,-1.,1.)
  # Preserve FP8 negative-zero bits explicitly in the independent decoder.
  values=values.astype(np.float32);values[(code&127)==0]=np.where(code[(code&127)==0]&128,np.float32(-0.),np.float32(0.))
  scale=scales[np.arange(n)//128].T[:,:,None]
  with np.errstate(all='ignore'):product=np.multiply(values,scale,dtype=np.float32)
  u=product.view(np.uint32);rounded=((u.astype(np.uint64)+0x7fff+((u>>16)&1))>>16).astype(np.uint16)
  want=rounded.transpose(1,0,2).reshape(n,g*128);want.tofile(folder/'cpu-rn32-rnbf16.bin')
  bits=u.transpose(1,0,2).reshape(n,g*128);finite32=(bits&0x7f800000)!=0x7f800000
  zeros=((results['old']&0x7fff)==0)&((want&0x7fff)==0)
  subnormal_or_zero=(bits&0x7f800000)==0
  bad=results['old']!=want
  sb=np.broadcast_to(scale.view(np.uint32),values.shape).transpose(1,0,2).reshape(n,g*128)
  scale_normal=((sb&0x7f800000)!=0)&((sb&0x7f800000)!=0x7f800000)
  scale_subnormal=((sb&0x7f800000)==0)&((sb&0x7fffffff)!=0)
  regular=finite32&~subnormal_or_zero&scale_normal
  assert not np.any(bad&regular),'normal scale/product independent CPU mismatch'
  record['cpu_diagnostic']=dict(finite_fp32_products=int(finite32.sum()),finite_product_mismatches=int((bad&finite32).sum()),signed_zero_only=int((bad&finite32&zeros).sum()),subnormal_or_zero_product_mismatches=int((bad&finite32&subnormal_or_zero).sum()),normal_finite_product_mismatches=int((bad&finite32&~subnormal_or_zero).sum()),subnormal_scale_and_normal_product_mismatches=int((bad&finite32&~subnormal_or_zero&scale_subnormal).sum()),normal_scale_normal_finite_product_mismatches=int((bad&regular).sum()),normal_scale_normal_finite_product_words=int(regular.sum()),scope='Independent native finite-code decode, RN32 product then BF16 RNE. CPU diagnostic only: common FTZ/zero and nonfinite behavior is retained separately, not silently accepted as IEEE arithmetic.')
 records.append(record);(out/'partial-result.json').write_text(json.dumps(records,indent=2)+'\n');assert mismatches==0,record
report=dict(status='PASS_ACTUAL_ELF_OLD_NEW_BITS',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),device_accessed=False,identities=identities,records=records,commands=commands,scope='Two actual deployed decoder libraries simulated; boundary equivalence, not device throughput or full-MAC acceptance.')
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(records,indent=2))
