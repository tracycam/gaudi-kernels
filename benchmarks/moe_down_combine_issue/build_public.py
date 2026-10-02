"""Package already-built actual ELFs, keeping a byte-identical old-body control."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--p4',type=Path,required=True);p.add_argument('--p6',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--include',default='/usr/lib/habanatools/include');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
meta={'device_used':False,'state':'BUILDING','elfs':{},'commands':[]}
try:
 for stem,path in [('down_p4',a.p4),('down_p6',a.p6),('down_old',a.old)]:
  dest=a.output/(stem+'.o');shutil.copy2(path,dest);meta['elfs'][stem]=hashlib.sha256(dest.read_bytes()).hexdigest()
  meta['commands'].append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',stem+'.o',stem+'_x86.o'])
 assert meta['elfs']['down_old']=='b930eb299830f036be389bb4a9d7f0597fa9ee24e51c2880e4e308689c6d3b75'
 meta['commands'].append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.include,str(root/'csrc/host/moe_down_combine_issue_glue.cpp'),'down_p4_x86.o','down_p6_x86.o','down_old_x86.o','-o','libgaudi_down_combine_isa_tpc.so'])
 with (a.output/'build.log').open('w') as log:
  for cmd in meta['commands']:log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=a.output,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
 meta.update(state='COMPILED_NOT_DEVICE_TESTED',library_sha256=hashlib.sha256((a.output/'libgaudi_down_combine_isa_tpc.so').read_bytes()).hexdigest())
except BaseException as error:meta.update(state='FAIL',error=repr(error));raise
finally:(a.output/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta))
