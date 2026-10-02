"""Build distinct folded GP/down libraries from pinned deployed embedded ELFs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from moe_activation_fold_core import embedded_elf,fold,sha,verify_deployed

p=argparse.ArgumentParser();p.add_argument('--gp-dir',required=True,type=Path);p.add_argument('--down-dir',required=True,type=Path)
p.add_argument('--output',required=True,type=Path);p.add_argument('--tpc-only',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
identity=root/'source-identity.json';commit=(json.loads(identity.read_text())['git_commit'] if identity.exists() and not (root/'.git').exists() else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
sources=['tools/build_moe_activation_folded.py','tools/moe_activation_fold_core.py','csrc/host/moe_activation_folded_glue.cpp','csrc/torch/moe_activation_folded.cpp']
for name in sources:
 if (root/'.git').exists():
  if subprocess.check_output(['git','show',commit+':'+name],cwd=root)!=(root/name).read_bytes():raise RuntimeError('commit source before building: '+name)
 else:
  if sha((root/name).read_bytes())!=json.loads(identity.read_text())['files_sha256'][name]:raise RuntimeError('source identity mismatch: '+name)
report={'status':'BUILDING','source_commit':commit,'committed_sources_verified':True,'device_acquired':False,'default_replaced':False,'commands':[],'targets':{}}
def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
def run(command):
 report['commands'].append(command);save()
 with (out/'commands.log').open('a') as log:
  log.write(json.dumps(command)+'\n');log.flush();subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
try:
 for target,directory in [('gp',a.gp_dir.resolve(strict=True)),('down',a.down_dir.resolve(strict=True))]:
  data,provenance=verify_deployed(directory,target);candidate,schedule=fold(data['assembly'].decode())
  stem='folded_'+target
  (out/(target+'_deployed.o')).write_bytes(data['elf']);(out/(target+'_deployed.s')).write_bytes(data['assembly'])
  (out/(stem+'.s')).write_text(candidate);(out/(stem+'.o')).write_bytes(data['elf'])
  run(['tpc-clang','-mcpu=gaudi2','-c',target+'_deployed.s','-o',target+'_deployed_slots.o'])
  run(['objcopy','--dump-section','.text='+target+'_deployed.text',target+'_deployed.o'])
  run(['objcopy','--dump-section','.text='+target+'_reassembled.text',target+'_deployed_slots.o'])
  if (out/(target+'_deployed.text')).read_bytes()!=(out/(target+'_reassembled.text')).read_bytes():raise ValueError('deployed text differs from supplied assembly')
  run(['tpc-clang','-mcpu=gaudi2','-c',stem+'.s','-o',stem+'_slots.o'])
  run(['objcopy','--dump-section','.text='+stem+'.text',stem+'_slots.o'])
  run(['objcopy','--update-section','.text='+stem+'.text',stem+'.o'])
  run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',stem+'.o',stem+'_x86.o'])
  with (out/(stem+'.objdump')).open('w') as log:subprocess.run(['tpc-llvm-objdump','--triple=tpc','-d',str(out/(stem+'.o'))],stdout=log,check=True,timeout=30)
  report['targets'][target]={'deployed_provenance':provenance,'schedule':schedule,'text_matches_deployed_assembly':True,
                            'candidate_elf_sha256':sha((out/(stem+'.o')).read_bytes()),'candidate_text_sha256':sha((out/(stem+'.text')).read_bytes())}
 glue=root/'csrc/host/moe_activation_folded_glue.cpp';(out/'glue.cpp').write_bytes(glue.read_bytes())
 run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include','glue.cpp','folded_gp_x86.o','folded_down_x86.o','-o','libgaudi_moe_activation_folded_tpc.so'])
 library=(out/'libgaudi_moe_activation_folded_tpc.so').read_bytes()
 for target in ['gp','down']:
  assert embedded_elf(library,'folded_'+target)==(out/('folded_'+target+'.o')).read_bytes()
 report['output_embedded_elfs_verified']=True
 if not a.tpc_only:
  os.environ.setdefault('PT_HPU_LAZY_MODE','1');os.environ.setdefault('MAX_JOBS','2')
  import torch
  import habana_frameworks.torch as ht
  from torch.utils.cpp_extension import load
  h=Path(ht.__file__).parent
  load(name='gaudi_moe_activation_folded',sources=[str(root/'csrc/torch/moe_activation_folded.cpp')],extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
  report['torch']=torch.__version__
 report.update(status='BUILT_NOT_DEVICE_QUALIFIED',files_sha256={p.name:sha(p.read_bytes()) for p in out.iterdir() if p.is_file() and p.name!='build.json'})
except Exception as error:report.update(status='FAIL',error=repr(error));raise
finally:save()
print(json.dumps({'status':report['status'],'output':str(out)}))
