"""Independent NumPy FP64 full-output oracle from original retained weight bytes."""
import argparse, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('case',type=Path);a=p.parse_args()
plan=next(json.loads(line) for line in (a.case/'run.log').read_text().splitlines() if line.startswith('{') and json.loads(line).get('stage')=='plan')
n,k=plan['N'],plan['K']; nb=(n+255)//256;ng=(k+31)//32
values=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float64)
checked=bad=0;error=energy=0.;maxabs=maxback=0.
for e,m in enumerate(map(int,plan['M'].split(','))):
    if not m:continue
    w=np.fromfile(a.case/f'e{e}_packed.bin',np.uint8).reshape(nb,k,128)
    scales=np.fromfile(a.case/f'e{e}_e8m0.bin',np.uint8).reshape(nb,ng,256)
    codes=np.concatenate((w&15,w>>4),axis=2).transpose(0,2,1).reshape(nb*256,k)[:n]
    exponents=np.repeat(scales.transpose(0,2,1).reshape(nb*256,ng)[:n],32,axis=1)[:,:k].astype(np.int32)-127
    weights=np.ldexp(values[codes],exponents)
    if plan['mode']=='decode':
        expected=weights.T.astype(np.float32).view(np.uint32)>>16
        actual=np.fromfile(a.case/f'e{e}_output.bin',np.uint16).reshape(k,n)
        bad+=np.count_nonzero(expected!=actual);checked+=actual.size;continue
    bits=np.fromfile(a.case/f'e{e}_activation.bin',np.uint16).astype(np.uint32)<<16
    x=bits.view(np.float32).reshape(m,k).astype(np.float64)
    ref=x@weights.T;absolute=np.abs(x)@np.abs(weights.T)
    if plan['bias']:
        bias=np.fromfile(a.case/f'e{e}_bias.bin',np.float32).astype(np.float64)
        ref+=bias;absolute+=np.abs(bias)
    actual=np.fromfile(a.case/f'e{e}_output.bin',np.float32).reshape(m,n)
    delta=np.abs(actual-ref);bad+=np.count_nonzero(~np.isfinite(actual)|(delta>2e-5+2e-6*absolute))
    checked+=actual.size;error+=np.sum(delta**2);energy+=np.sum(ref**2)
    maxabs=max(maxabs,float(delta.max()));maxback=max(maxback,float((delta/np.maximum(absolute,1e-30)).max()))
    saved=np.fromfile(a.case/f'e{e}_oracle_f64.bin',np.float64).reshape(m,n)
    assert np.allclose(saved,ref,rtol=1e-12,atol=1e-12),'C++ oracle differs from independent NumPy oracle'
record={'case':a.case.name,'checked':checked,'bad':int(bad),'relative_l2':float(np.sqrt(error/max(energy,1e-30))),
        'max_abs':maxabs,'max_componentwise_backward':maxback,'oracle':'NumPy FP64 original nibbles and E8M0 + BF16 activation'}
print(json.dumps(record));raise SystemExit(1 if bad else 0)
