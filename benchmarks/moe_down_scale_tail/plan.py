"""Pure host fixture binding and complete ABBA accounting; never imports Torch."""
import hashlib,json,math,sys
from pathlib import Path
ARMS=('deployed289','candidate242','candidate242','deployed289')
STATES=('initial','hot_changed','cold_changed')
PINS={
 'gp_tpc':('production-runtime/gp-scale-tail/tpc.so','f45f038808d6ec6ba14ac99445fe7bdce35a523c6f63271a3f37624e57130e07'),
 'gp_torch':('production-runtime/gp-scale-tail/torch.so','c73a48e86e32fb5b531cc97d5c205bc35f28c654820773cd82814702688db410'),
 'batch_tpc':('batch-kernel-source/tpc/libbatch_tpc.so','e9447e75356498df226b7d7daa3c2dac3f20efbab32834fc030000e61bdb68e2'),
 'batch_torch':('batch-kernel-source/ops-build/unified_batch_ops.so','fd95be3c7dc08d099489fe1dbe88498a8109fbbc43387c263c85e17cb8f9d159'),
 'precision_tpc':('precision-source/precision-tpc/libprecision_tpc.so','f3e6f643eff6482218c8c551b112df8f3e50740ca88a7e7200daae005015d1dc'),
 'precision_torch':('precision-source/precision-ops-build/precision_ops.so','e6990cef2ea29e7d061e0bc1d06949af757b3457ee55150db325bf94cd4f4fae'),
 'legacy_down_tpc':('production-runtime/libraries/libgaudi_down_activation_tpc.so','9675942ebd61768cfa16064f9a034ef34f11e42238b0dcdfdca1a5811d73c47a'),
 'legacy_down_torch':('production-runtime/libraries/gaudi_down_activation.so','8774a7107f1242a7d5e3626b0166cbac020cb34b47e488c566927f695de08f1d')}
ELF_PINS={'control':'b930eb299830f036be389bb4a9d7f0597fa9ee24e51c2880e4e308689c6d3b75','candidate':'d439656a16fa2d913673dfeb90a3407af686a827ec4b2bfaa64295bee978fd3b'}
WEIGHTS_SHA='fadd659b0343bb2b19928c137e7a531899593f4275c206930a5e627015f68a87'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate(root):
 root=Path(root).resolve(strict=True);plan=json.loads((root/'plan.json').read_text());assert plan['format']=='gk-down-scale-tail-full-chain-v1'
 for name,record in plan['files'].items():
  p=(root/record['path']).resolve(strict=True);assert p.is_relative_to(root)and sha(p)==record['sha256'],name
 for key,(_,digest)in PINS.items():assert plan['files'][key]['sha256']==digest,key
 assert plan['files']['weights']['sha256']==WEIGHTS_SHA
 assert plan['embedded_elfs']==ELF_PINS
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
 from moe_activation_fold_core import embedded_elf
 binary=(root/plan['files']['dual_down_tpc']['path']).read_bytes()
 for name,digest in ELF_PINS.items():assert hashlib.sha256(embedded_elf(binary,name)).hexdigest()==digest,name
 return plan
def schedule(rows,trials):
 assert rows and len(set(rows))==len(rows)and all(type(n)is int and n in(1,2,8)for n in rows)
 assert type(trials)is int and 1<=trials<=5
 return[dict(rows=n,state=state,trial=t,arm_index=i,variant=arm)for n in rows for state in STATES for t in range(trials)for i,arm in enumerate(ARMS)]
def complete(records,rows,trials):
 expected=schedule(rows,trials);assert len(records)==len(expected),'missing timed arms; no truncated acceptance'
 for got,want in zip(records,expected):
  assert all(got.get(k)==v for k,v in want.items()),(got,want)
  assert got['output_bits_equal']and got['stable_logical_handles']
  assert all(math.isfinite(got[k])and got[k]>0 for k in('event_us','wall_us'))
  assert .5<got['event_us']/got['wall_us']<1.2,'event/wall crosscheck failed'
 return True
