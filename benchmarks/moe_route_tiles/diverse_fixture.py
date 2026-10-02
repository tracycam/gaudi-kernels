"""E384 exact-GP fixture with distinct original-width expert row permutations.

Every GP dot is a permutation of a certified exact FP32 dot. The literal
elementwise gate is permuted identically, avoiding a CPU exp approximation.
Down references provide FP32 outputs and absolute-product error bounds.
This is synthetic coverage, not checkpoint/model quality evidence.
"""
import argparse, hashlib, importlib.util, json, os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,default=513);a=p.parse_args()
assert 1<=a.tokens<=513
source=a.input.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('historical_packing',root/'csrc/tpc/mxfp4_moe/legacy/packing.py');packing=importlib.util.module_from_spec(spec);spec.loader.exec_module(packing)
old=json.loads((source/'fixture.json').read_text());assert old['T']==old['R']==old['E']==2 and old['case']=='mixed' and old['GP_full_FP64_exactly_representable_in_FP32']
e,t,r=384,a.tokens,8
gp=np.lib.format.open_memmap(out/'gp.npy',mode='w+',dtype=np.uint8,shape=(e*6144,256))
gs=np.lib.format.open_memmap(out/'gps.npy',mode='w+',dtype=np.uint8,shape=(e*192,512))
down=np.lib.format.open_memmap(out/'down.npy',mode='w+',dtype=np.uint8,shape=(e*3072,256))
ds=np.lib.format.open_memmap(out/'downs.npy',mode='w+',dtype=np.uint8,shape=(e*96,512))
basew=[np.load(source/f'gp-e{i}-original.npy')for i in range(2)]
bases=[np.load(source/f'gp-e{i}-original-scales.npy')for i in range(2)]
bd=np.load(source/'down.npy').reshape(2,3072,256);bs=np.load(source/'downs.npy').reshape(2,96,512)
original_gate=np.load(source/'gate_bits.npy')
gates=np.empty((e,2,256),np.float64)
for expert in range(e):
    base=expert%2;shift=expert//2;indices=(np.arange(256)+shift)%256
    perm=np.concatenate((indices,indices+256));w,s=packing.pack(basew[base][perm],bases[base][perm])
    wr,sr=packing.unpack(w,s);assert np.array_equal(wr,basew[base][perm])and np.array_equal(sr,bases[base][perm])
    gp[expert*6144:(expert+1)*6144]=w.reshape(6144,256);gs[expert*192:(expert+1)*192]=s.reshape(192,512)
    down[expert*3072:(expert+1)*3072]=bd[base];ds[expert*96:(expert+1)*96]=bs[base]
    gates[expert]=((original_gate[:,base,indices].astype(np.uint32)<<16).view(np.float32)).astype(np.float64)
for owner in (gp,gs,down,ds):owner.flush()
values=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float64)
dots=np.empty((e,2,6144),np.float64);absolute=np.empty_like(dots)
for base in range(2):
    raw=np.load(source/f'down-e{base}-original.npy');sc=np.load(source/f'down-e{base}-original-scales.npy')
    q=np.empty((6144,256),np.uint8);q[:,::2]=raw&15;q[:,1::2]=raw>>4
    w=values[q]*np.exp2(np.repeat(sc.astype(np.int32)-127,32,axis=1))
    g=gates[base::2].reshape(-1,256)
    dots[base::2]=(g@w.T).reshape(e//2,2,6144)
    absolute[base::2]=(np.abs(g)@np.abs(w).T).reshape(e//2,2,6144)
np.save(out/'expert_down_exact.npy',dots);np.save(out/'expert_down_absolute.npy',absolute)
xbase=np.load(source/'x.npy');states=[]
for mode in ('uniform','hot','skew','zero'):
    ids=np.array([[(row*r+slot)%e for slot in range(r)]for row in range(t)],np.int32)
    row_index=np.arange(t)%2
    if mode in ('hot','skew'):
        ids[:t if mode=='hot'else 3*t//4]=np.arange(e-r,e,dtype=np.int32)[::-1]
        row_index=1-row_index
    routing=np.full((t,r),.125,np.float32)
    if mode=='zero':routing.fill(0)
    expected=np.zeros((t,6144),np.float32);bound=np.zeros((t,6144),np.float64)
    for slot in range(r):
        selected=dots[ids[:,slot],row_index]
        expected=(selected.astype(np.float32).astype(np.float64)*routing[:,slot,None]+expected.astype(np.float64)).astype(np.float32)
        bound+=absolute[ids[:,slot],row_index]*np.abs(routing[:,slot,None])
    state=out/mode;state.mkdir()
    for name,data in [('x',xbase[row_index]),('ids',ids),('routing',routing),('expected',expected),('absolute',bound)]:np.save(state/(name+'.npy'),data)
    states.append(mode)
for name in ('x','ids','routing','expected','absolute'):os.link(out/'uniform'/(name+'.npy'),out/(name+'.npy'))
os.link(source/'lut.npy',out/'lut.npy')
provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in source.glob('*.npy')}
meta=dict(E=e,T=t,R=r,case='diverse_permuted',layout='HistoricalN512',states=states,
    GP_full_FP64_exactly_representable_in_FP32=True,
    gate_oracle='Permutation of actual literal TPC ISA gate outputs on exact-GP base rows',
    expert_identity='base=expert%2, shift=expert//2; rotate both gate/up N halves identically, leave down K unchanged',
    down_oracle='Original-byte FP64 dot used for FP32 gamma bound, not exact-output equality',
    source_fixture=str(source),source_files_sha256=provenance,device_validated=False,
    scope='Distinct E384 synthetic expert owners and uniform/hot/skew routes; not checkpoint/model qualification')
(out/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(dict(T=t,E=e,R=r,output=str(out))),flush=True)
