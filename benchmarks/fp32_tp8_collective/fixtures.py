"""Original FP32 bytes, FP64 oracle, and ordered FP32 precision_ops reference."""
import argparse,json
from pathlib import Path
import numpy as np

def create(kind,stages=69):
    rng=np.random.default_rng(987321)
    lane=np.arange(6144)[None,None,:];rank=np.arange(8)[:,None,None];stage=np.arange(stages)[None,:,None]
    if kind=='exact':x=((lane%13)-6+rank*3+(stage%7)).astype(np.float32)
    elif kind=='ordinary':
        x=(rng.normal(size=(8,stages,6144))*np.exp2(rng.integers(-12,5,size=(8,stages,6144)))).astype(np.float32)
        # Full FP32 lanes deliberately differ from BF16; cancellation/order is
        # measured relative to FP64, not required to equal one chosen AR tree.
        x[:, :, ::8]=np.array([2**24,1.,-2**24,1.,.25,-.125,.0625,-.03125],np.float32)[:,None,None]
        x[:, :, 1::8]=np.array([1.,2**-10,2**-12,-.5,2**-18,-2**-19,2**-20,0],np.float32)[:,None,None]
    else:raise ValueError(kind)
    oracle=x.astype(np.float64).sum(axis=0);absolute=np.abs(x.astype(np.float64)).sum(axis=0)
    ordered=np.zeros_like(x[0])
    for r in range(8):ordered=(ordered+x[r]).astype(np.float32)
    return x,oracle,absolute,ordered

def metric(x,oracle,absolute):
    delta=np.abs(x.astype(np.float64)-oracle);gamma=7*np.finfo(np.float32).eps/(1-7*np.finfo(np.float32).eps)
    bad=~np.isfinite(x)|(delta>gamma*absolute)
    return dict(checked=x.size,bad=int(bad.sum()),max_abs=float(delta.max()),relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(oracle),np.finfo(float).tiny)),max_backward_error=float(np.max(delta/np.maximum(absolute,np.finfo(float).tiny))),nonzero_reference=int(np.count_nonzero(oracle)),unexpected_zero=int(np.count_nonzero((oracle!=0)&(x==0))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[]
    for kind in ['exact','ordinary']:
        d=a.output/kind;d.mkdir();x,ref,ab,ordered=create(kind)
        for rank in range(8):x[rank].tofile(d/f'input-rank{rank}.bin')
        x.tofile(d/'all-inputs.bin');ref.tofile(d/'oracle-fp64.bin');ab.tofile(d/'absolute-fp64.bin');ordered.tofile(d/'ordered-fp32.bin')
        result=metric(ordered,ref,ab);assert result['bad']==0
        if kind=='exact':assert np.array_equal(ordered.astype(np.float64),ref)
        bits=x.view(np.uint32);bf16=((bits+0x7fff+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
        info=dict(non_BF16_values=int(np.count_nonzero(x!=bf16)),kind=kind,shape=[8,69,6144],dtype='float32',send_count=6144,send_bytes=24576,gather_output_bytes=196608,**result)
        (d/'fixture.json').write_text(json.dumps(info,indent=2)+'\n');records.append(info)
    (a.output/'cpu-check.json').write_text(json.dumps(dict(all_pass=True,device_validated=False,records=records),indent=2)+'\n')
