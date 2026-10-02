"""Build actual-ELF AV edit; source text and debug stores preserve other packets."""
import argparse,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(a.input)=='08ee05f4b8bb399b04c6861ed4d0c5db720c7f168f968052f9f9e0314385ac66'
shutil.copy2(a.input,out/'template.o');meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'commands':[],'device_verified':False,'status':'BUILDING'}
with(out/'build.log').open('w')as log:
 def run(cmd,capture=None):
  meta['commands'].append(cmd);log.write(json.dumps(cmd)+'\n');log.flush()
  if capture:
   with(out/capture).open('w')as f:subprocess.run(cmd,cwd=out,stdout=f,stderr=log,check=True)
  else:subprocess.run(cmd,cwd=out,stdout=log,stderr=log,check=True)
 try:
  run(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn','template.o'],'template.dis')
  run([sys.executable,str(root/'benchmarks/swa128_avgroup/patch_av.py'),'template.dis','--output','assembly'])
  for name in ['quad','group','quad_debug','group_debug']:
   run(['tpc-clang','-x','assembler','-mcpu=gaudi2','-c','assembly/'+name+'.s','-o',name+'_slots.o'])
   run(['objcopy','--dump-section','.text='+name+'.text',name+'_slots.o',name+'_extract.o'])
   run(['objcopy','--update-section','.text='+name+'.text','template.o',name+'.o'])
   run([sys.executable,str(root/'benchmarks/swa128_ilp/fix_symbols.py'),name+'.o',name+'_slots.o'])
   run(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',name+'.o'],name+'.dis')
  sys.path.insert(0,str(root/'benchmarks/swa128_avgroup'))
  from patch_av import parse
  for name in ['quad','group','quad_debug','group_debug']:
   expected=[re.sub(r'\s+',' ',s.strip())for s in (out/'assembly'/(name+'.s')).read_text().splitlines()[2:]]
   actual=[re.sub(r'\s+',' ',s.strip())for s in parse(out/(name+'.dis'))]
   assert actual==expected,[(i,x,y)for i,(x,y)in enumerate(zip(expected,actual))if x!=y][:5]
  meta['assembler_packet_semantics_readback_exact']=True
  run(['objcopy','--dump-section','.text=template.text','template.o','template_extract.o'])
  assert(out/'template.text').read_bytes()==(out/'quad.text').read_bytes(),'unmodified assembler roundtrip changed instruction bytes'
  meta['roundtrip_text_bit_exact']=True
  run(['g++','-O2','-std=c++17','-I/usr/lib/habanatools/include',str(root/'benchmarks/swa128_avgroup/simulator.cpp'),'-L/usr/lib/habanatools','-ltpc_tests_core_ext','-Wl,-rpath,/usr/lib/habanatools','-o','simulator'])
  meta['status']='COMPILED'
 finally:
  meta['files_sha256']={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()}
  (out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
