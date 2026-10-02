"""Build an isolated fixed-activation diagnostic from the measured legacy ELF.

All changed MAC operands are BF16 immediate one; activation loads/address sites
become NOPs. The unchanged packet locations/MAC order test activation-dependency
cost. This is invalid for arbitrary activation values and is never installed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--fixture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--variant',choices=['imm1','broadcast'],default='imm1')
p.add_argument('--target',choices=['gp','down'],default='gp',help='down reuses only the proven generic K32 activation schedule')
p.add_argument('--production-guids',action='store_true',help='broadcast only: retain all seven original ub_* GUIDs for isolated replacement library')
p.add_argument('--tpc-only',action='store_true',help='skip experimental Torch extension build')
a=p.parse_args();fixture=a.fixture.resolve(strict=True);out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
if a.production_guids and a.variant!='broadcast':p.error('constant-one diagnostic must never use production GUIDs')
if a.target=='down' and a.variant!='broadcast':p.error('down only supports the general broadcast candidate')
root=Path(__file__).resolve().parents[1]
snapshot=json.loads((fixture/'snapshot.json').read_text())
for name,sha in snapshot['files_sha256'].items():
    assert hashlib.sha256((fixture/name).read_bytes()).hexdigest()==sha,name
meta={'status':'BUILDING','scope':'fixed BF16-one activation diagnostic only; no default change',
      'fixture_manifest_sha256':hashlib.sha256((fixture/'snapshot.json').read_bytes()).hexdigest(),
      'device_acquired':False,'commands':[]}
identity=root/'source-identity.json'
meta['source_commit']=(json.loads(identity.read_text())['git_commit'] if identity.exists() and not (root/'.git').exists()
                       else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
def save():(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
def run(cmd):
    meta['commands'].append(cmd);save();subprocess.run(cmd,cwd=out,check=True)
try:
    names=['direct_gp','compact_gate','direct_down','gate_broadcast','sorted_prep','sorted_combine','masked_gemv']
    for name in names:shutil.copyfile(fixture/'tpc'/f'{name}.o',out/f'{name}.o')
    target='direct_'+a.target
    s=(fixture/'tpc'/(target+'.s')).read_text()
    if a.variant=='imm1':
        s,n=re.subn(r'(mac\.bf16 acc_fp32 D\d+, V\d+, )S(?:16|17|18|19|20)',r'\g<1>0x3f80',s);assert n==128
        s,nload=re.subn(r'ld_g S(?:16|17|18|19|20), AD[012]','nop',s);assert nload==32
        s,naddr=re.subn(r'gen_addr dt=int8 AD[012], 0x2, I6','nop',s);assert naddr==32
        meta['changes']={'mac_operands':n,'activation_loads':nload,'activation_addresses':naddr}
    else:
        from moe_gp_broadcast_schedule import transform
        s,details=transform(s)
        meta.update(scope=f'general BF16 K32 tensor-fetch/broadcast {a.target} candidate; original MAC/scale order',changes=details)
    stem=target+'_'+a.variant
    (out/(stem+'.s')).write_text(s)
    run(['tpc-clang','-mcpu=gaudi2','-c',stem+'.s','-o',stem+'_slots.o'])
    run(['objcopy','--dump-section','.text='+stem+'.text',stem+'_slots.o'])
    run(['objcopy','--update-section','.text='+stem+'.text',target+'.o'])
    glue=(fixture/'glue.cpp').read_text()
    if not a.production_guids:
        prefix='diag_' if a.target=='gp' else 'downact_'
        for name in names:glue=glue.replace('"ub_'+name+'"','"'+prefix+name+('_'+a.variant if name==target else '')+'"')
    meta['production_guids']=a.production_guids
    (out/'glue.cpp').write_text(glue)
    for name in names:run(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
    library='gaudi_gp_diagnostic' if a.target=='gp' else 'gaudi_down_activation'
    run(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',
         'glue.cpp',*[name+'_x86.o' for name in names],'-o','lib'+library+'_tpc.so'])
    if not a.tpc_only:
        os.environ.setdefault('PT_HPU_LAZY_MODE','1');os.environ.setdefault('MAX_JOBS','2')
        import torch
        import habana_frameworks.torch as ht
        from torch.utils.cpp_extension import load
        h=Path(ht.__file__).parent
        source='moe_gp_diagnostic.cpp' if a.target=='gp' else 'moe_down_broadcast.cpp'
        load(name=library,sources=[str(root/'csrc/torch'/source)],
             extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],
             extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],
             build_directory=str(out),is_python_module=False,verbose=True)
        meta['torch']=torch.__version__
    meta.update(status='BUILT_NOT_DEVICE_VALIDATED',files_sha256={
        p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='build.json'})
except Exception as error:meta.update(status='FAIL',error=repr(error));raise
finally:save()
