"""CPU proof checks for the metadata-only predispatch bound (no HPU calls)."""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np
from verify import Native

def safe(a, scale_min, scale_max, n, k, bias=(), force=False):
    if force or n%512 or k%32 or not 2<=scale_min<=scale_max<=252 or k+1>1048576:
        return False
    exponents=[((int(x)>>7)&255) or 1 for x in a if int(x)&32767]
    if 255 in exponents:return False
    low,high=1000,-1000
    if exponents:
        lo,hi=min(exponents),max(exponents)
        if lo<9 or hi>245:return False
        low,high=lo+scale_min-262,hi+scale_max-251
    for raw in bias:
        raw=int(raw);e=(raw>>23)&255;m=raw&0x7fffff
        if e==255:return False
        if e:m|=0x800000
        else:e=1
        if m:
            low=min(low,e-150)
            high=max(high,e-127)
    ceil_log2=k.bit_length() # ceil(log2(K+1)), for K>=1
    return low>=-126 and high+ceil_log2<=126

def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--old-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    native=Native(args.library);accepted=0;checked=0
    for ae in range(1,255):
        a=[(ae<<7)|127]*32
        for scale in range(255):
            admitted=safe(a,scale,scale,512,32)
            if admitted:
                assert native.call('fp32_safe',a,[7]*32,[scale]*32)==1
                accepted+=1
            checked+=1
    old_cases=[]
    for case,n,k in [('n512-k6144-m1-history-v6',512,6144),('n6144-k256-m1-history-v6',6144,256),('stream-history-v6',32768,6144),('guarded-subnormal-v8',512,256),('exact512-subnormal-mixed-v8',513,257)]:
        d=args.old_root/case
        a=np.fromfile(d/'e0_activation.bin',np.uint16)[:k]
        scales=np.fromfile(d/'e0_e8m0.bin',np.uint8)
        admitted=safe(a,int(scales.min()),int(scales.max()),n,k)
        expected=case.endswith('history-v6')
        assert admitted==expected
        old_cases.append({'case':case,'fast_admitted':admitted,'scale_min':int(scales.min()),'scale_max':int(scales.max())})
    copy_boundaries=[]
    for k,nblocks,splits in [(65536,24,1),(131072,24,1),(65568,24,1),(65536,1,16)]:
        kp=k//splits;source=np.arange(k,dtype=np.uint16);copied=np.empty((splits,kp),np.uint16)
        for split in range(splits):
            for off in range(0,kp,128):
                start=split*kp+off;remaining=min(k-start,kp-off,128)
                copied[split,off:off+remaining]=source[start:start+remaining]
        assert np.array_equal(copied.reshape(-1),source)
        copy_boundaries.append({'K':k,'Nblocks':nblocks,'splits':splits,'copied_bits_equal':True})
    result={'activation_copy_boundaries':copy_boundaries,'status':'CPU_guard_only_no_device_validation','all_pass':True,'exponent_scale_pairs_checked':checked,'pairs_admitted':accepted,'predicate':'normal product/unscaled lattices plus rounded sum bound, ordinary FP32 rounding allowed','representative_saved_inputs':old_cases}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__ == "__main__":
    main()
