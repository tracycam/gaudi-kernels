"""Freeze existing production sources and qualified libraries; never rebuild."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from moe_activation_fold_core import embedded_elf

p=argparse.ArgumentParser();p.add_argument('--frozen-source',type=Path,required=True)
p.add_argument('--folded-dir',type=Path,required=True);p.add_argument('--down-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=a.frozen_source.resolve(strict=True);out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
frozen=json.loads((source/'sha256.json').read_text())
mapping={name:'executor/'+name for name in ['batch_ops.py','native_ops.py','precision_ops.py','kernels/packing.py',
    'torch-build/native_mxfp4_ops.so','tpc/libnative_tpc.so']}
mapping.update({'batch-kernel-source/ops-build/unified_batch_ops.so':'ops-build/unified_batch_ops.so',
                'batch-kernel-source/tpc/libbatch_tpc.so':'tpc/libbatch_tpc.so',
                'precision-source/precision-ops-build/precision_ops.so':'precision-ops-build/precision_ops.so',
                'precision-source/precision-tpc/libprecision_tpc.so':'precision-tpc/libprecision_tpc.so'})
files={}
def copy(path,name,expected):
    data=path.read_bytes();digest=hashlib.sha256(data).hexdigest();assert digest==expected,(path,digest)
    target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    files[name]={'sha256':digest,'bytes':len(data),'source':str(path.resolve())}
for name,target in mapping.items():copy(source/name,target,frozen[name])
pins={'folded':{'gaudi_moe_activation_folded.so':'54529adf89e7d1527df03ffbfb71c5047d518f406ed4513c10691108fbc50e78',
                'libgaudi_moe_activation_folded_tpc.so':'dd913322ff1d8f2f665afef1b50c28e53f9bedcaca85ca49723434073421614b'},
      'down':{'gaudi_down_activation.so':'8774a7107f1242a7d5e3626b0166cbac020cb34b47e488c566927f695de08f1d',
              'libgaudi_down_activation_tpc.so':'9675942ebd61768cfa16064f9a034ef34f11e42238b0dcdfdca1a5811d73c47a'}}
for group,directory in [('folded',a.folded_dir),('down',a.down_dir)]:
    for name,digest in pins[group].items():copy(directory/name,'libraries/'+group+'/'+name,digest)
assert 'def set_gp_policy' in (out/'executor/batch_ops.py').read_text()
elfs={}
for name,library,stem in [('broadcast_gemv','executor/tpc/libnative_tpc.so','gemv'),
                          ('folded_gp','libraries/folded/libgaudi_moe_activation_folded_tpc.so','folded_gp'),
                          ('vector_down','libraries/down/libgaudi_down_activation_tpc.so','direct_down')]:
    data=embedded_elf((out/library).read_bytes(),stem);target=out/'isa'/(name+'.o');target.parent.mkdir(exist_ok=True);target.write_bytes(data)
    with target.with_suffix('.dis').open('w')as log:
        subprocess.run(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',str(target)],stdout=log,stderr=log,check=True)
    elfs[name]={'library':library,'stem':stem,'sha256':hashlib.sha256(data).hexdigest()}
for path in (out/'isa').iterdir():files[str(path.relative_to(out))]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'source':'extracted actual embedded ELF, no recompilation'}
plan={'format':'gk-moe-compact8-runtime-v1','frozen_source':str(source),'files':files,'embedded_elfs':elfs,
      'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=Path(__file__).resolve().parents[1]).strip(),
      'torch_libraries':['executor/torch-build/native_mxfp4_ops.so','ops-build/unified_batch_ops.so','precision-ops-build/precision_ops.so',
                         'libraries/folded/gaudi_moe_activation_folded.so','libraries/down/gaudi_down_activation.so'],
      'tpc_libraries':['executor/tpc/libnative_tpc.so','tpc/libbatch_tpc.so','precision-tpc/libprecision_tpc.so',
                       'libraries/folded/libgaudi_moe_activation_folded_tpc.so','libraries/down/libgaudi_down_activation_tpc.so'],
      'device_verified':False,'recompiled':False}
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
