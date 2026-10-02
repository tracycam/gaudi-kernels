"""Package already-simulated ELF bytes under isolated AV candidate GUIDs."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--include',default='/usr/lib/habanatools/include');a=p.parse_args()
root=Path(__file__).resolve().parents[1];build=a.build.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
meta=json.loads((build/'build.json').read_text());assert meta['status']=='COMPILED'and meta['roundtrip_text_bit_exact']and meta['assembler_packet_semantics_readback_exact']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'input_build':str(build),'input_build_sha256':sha(build/'build.json'),'commands':[],'state':'BUILDING'}
with(out/'build.log').open('w')as log:
 def run(cmd):r['commands'].append(cmd);log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=out,stdout=log,stderr=log,check=True)
 try:
  names=['quad','group','quad_debug','group_debug']
  for n in names:
   assert sha(build/(n+'.o'))==meta['files_sha256'][n+'.o'];shutil.copy2(build/(n+'.o'),out/(n+'.o'))
   run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',n+'.o',n+'_x86.o'])
  run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.include,str(root/'csrc/host/swa128_avgroup_glue.cpp'),*[n+'_x86.o'for n in names],'-o','libgaudi_swa128_avgroup_tpc.so'])
  r['state']='BUILT_IDENTICAL_SIM_ELFS'
 finally:
  r['files_sha256']={p.name:sha(p)for p in out.iterdir()if p.is_file()};(out/'build.json').write_text(json.dumps(r,indent=2)+'\n')
