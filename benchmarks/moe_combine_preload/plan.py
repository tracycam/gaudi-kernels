"""Host-only immutable fixture/library binding and complete combine ABBA plan."""
import hashlib,importlib.util,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('gk_original_down_fixture',ROOT/'benchmarks/moe_down_scale_tail/plan.py')
_fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(_fixture)
ARMS=('deployed','unroll8','unroll8','deployed')
STATES=('initial','hot_changed','cold_changed')
VARIANTS=('deployed','control','unroll8')
TPC_SHA256='5e879072171dd1bcf2fbc610e05048937d4ac2559f0d9792fb56073fc6136640'
ELF_PINS={'control':'bccde2def546e9d0b68e9f5fce6def7af382bf5589b7980fa713845aa2fa3ff6',
          'unroll8':'59b250ba14084ac21a47798c37292deb61f8c5ad45fc83161549816e37cb3b30'}
sha=_fixture.sha

def validate(runtime):
    # Keep EVERY previous file, weight, TPC/Torch and embedded-down identity
    # check, even for the immutable fixture's unused experimental down library.
    original=_fixture.validate(runtime)
    result=json.loads(json.dumps(original))
    assert set(original['tpc_order'])=={'gp_tpc','batch_tpc','precision_tpc','legacy_down_tpc','dual_down_tpc'}
    assert set(original['torch_order'])=={'gp_torch','batch_torch','precision_torch','legacy_down_torch'}
    result['tpc_order']=[k for k in original['tpc_order'] if k!='dual_down_tpc']
    result['experiment']='fixed deployed289 down; combine-only comparison'
    return result

def validate_combine(path):
    path=Path(path).resolve(strict=True)
    assert sha(path)==TPC_SHA256,'unqualified combine database bytes'
    sys.path.insert(0,str(ROOT/'tools'))
    from moe_activation_fold_core import embedded_elf
    blob=path.read_bytes()
    for name,digest in ELF_PINS.items():assert hashlib.sha256(embedded_elf(blob,name)).hexdigest()==digest,name
    return dict(path=str(path),sha256=TPC_SHA256,embedded_elfs=ELF_PINS)

def schedule(rows,trials):
    assert rows and len(set(rows))==len(rows)and all(type(n)is int and n in(1,2,8)for n in rows)
    assert type(trials)is int and 1<=trials<=5
    return[dict(rows=n,state=state,trial=t,arm_index=i,variant=v)
           for n in rows for state in STATES for t in range(trials)for i,v in enumerate(ARMS)]

def complete(records,rows,trials):
    expected=schedule(rows,trials)
    assert len(records)==len(expected),'missing timed arms; no truncated acceptance'
    for got,want in zip(records,expected):
        assert all(got.get(k)==v for k,v in want.items()),(got,want)
        assert got['output_bits_equal']and got['stable_logical_handles']
        assert all(math.isfinite(got[k])and got[k]>0 for k in('event_us','wall_us'))
        assert .5<got['event_us']/got['wall_us']<1.2,'event/wall crosscheck failed'
        assert len(got['output_sha256'])==2 and all(len(x)==64 for x in got['output_sha256'])
    return True
