"""Offline freeze of exact production GP/gate/combine and original-width fixture."""
import argparse,json,shutil,sys
from pathlib import Path
from plan import PINS,ELF_PINS,WEIGHTS_SHA,sha
p=argparse.ArgumentParser();p.add_argument('--frozen-source',type=Path,required=True);p.add_argument('--down-tpc',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'));from moe_activation_fold_core import embedded_elf
source=a.frozen_source.resolve(strict=True);out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source_pins=json.loads((source/'sha256.json').read_text());files={}
def copy(key,src,name,digest):
 assert sha(src)==digest,(key,sha(src),digest);dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(dst)==digest
 files[key]=dict(path=name,sha256=digest,source=str(src.resolve()),bytes=dst.stat().st_size)
for key,(name,digest)in PINS.items():
 assert source_pins[name]==digest;copy(key,source/name,'libraries/'+key+'.so',digest)
for name in('batch_ops.py','native_ops.py','precision_ops.py','kernels/packing.py'):
 copy('source_'+name,source/name,'source/'+name,source_pins[name])
copy('weights',a.weights,'prepared-weights.pt',WEIGHTS_SHA)
copy('dual_down_tpc',a.down_tpc,'libraries/dual_down_tpc.so',sha(a.down_tpc))
for stem,digest in ELF_PINS.items():
 data=embedded_elf(a.down_tpc.read_bytes(),stem);import hashlib
 assert hashlib.sha256(data).hexdigest()==digest
 dest=out/'isa'/(stem+'.o');dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
plan=dict(format='gk-down-scale-tail-full-chain-v1',files=files,embedded_elfs=ELF_PINS,tpc_order=['gp_tpc','batch_tpc','precision_tpc','legacy_down_tpc','dual_down_tpc'],torch_order=['gp_torch','batch_torch','precision_torch','legacy_down_torch'],E=24,R=8,GP=[512,6144],down=[6144,256],device_tested=False,scope='Immutable production arithmetic and 60,162,048B original-width synthetic weights; no expanded HBM cache or model acceptance')
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
