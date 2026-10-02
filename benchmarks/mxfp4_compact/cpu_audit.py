"""Offline layout/read-schedule audit. This does NOT execute or certify TPC lanes."""
import argparse
import ctypes
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python'))
from gaudi_kernels.mxfp4_compact import PreparedCompact, prepare, unpack, from_buffers
from gaudi_kernels.mxfp4_prepared_gemv import pack as pack_v2

LUT = np.array([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])


def read_schedule(p):
    """Declared memory-element requests of decode_native.c / decode_rows.c.

    (blob, start, stop, region). Output/LUT/activation/descriptor reads are not
    weight reads and are NOT hidden in this ledger. Physical HBM transactions
    and cache reuse cannot be established without native profiling.
    """
    nf, kf, groups = p.n_full, p.k_full, p.k_full//32
    for block in range(nf//512):
        for g in range(groups):
            for pair in range(2):
                for half in range(2):
                    start=block*groups*512+g*512+pair*256+half*128
                    yield 'scale',start,start+128,'native_n512'
                for k in range(g*32,(g+1)*32):
                    start=block*kf*256+k*256+pair*128
                    yield 'weight',start,start+128,'native_n512'
    for n in range(p.n-nf):
        for g in range(groups):
            start=nf*kf//2+n*kf//2+g*16
            yield 'weight',start,start+16,'n_tail_rows'
            start=nf*groups+n*groups+g
            yield 'scale',start,start+1,'n_tail_rows'
    if p.k>kf:
        width=(p.k-kf+1)//2
        for n in range(p.n):
            start=p.n*kf//2+n*width
            yield 'weight',start,start+width,'k_tail_rows'
            start=p.n*groups+n
            yield 'scale',start,start+1,'k_tail_rows'


def audit_intervals(p):
    intervals=list(read_schedule(p))
    result={}
    for blob,size in [('weight',p.weight.size),('scale',p.scales.size)]:
        spans=sorted((start,stop,region) for b,start,stop,region in intervals if b==blob)
        cursor=0
        for start,stop,region in spans:
            assert start==cursor and stop>start, (blob,cursor,start,stop,region)
            cursor=stop
        assert cursor==size
        result[blob]={'bytes':size,'requests':len(spans),
                      'request_size_histogram':dict(sorted(Counter(stop-start for start,stop,_ in spans).items())),
                      'exactly_once':True}
    return result,intervals


def schedule_decode(p):
    """CPU model of decoded logical values from each disjoint input segment.

    This expresses intended native even/odd->linear and tail nibble interleave.
    It is not an emulator for the compiled TPC instructions.
    """
    codes=np.empty((p.n,p.k),np.uint8)
    exponents=np.empty((p.n,(p.k+31)//32),np.uint8)
    nf,kf,g=p.n_full,p.k_full,p.k_full//32
    for block in range(nf//512):
        w=p.weight[block*kf*256:(block+1)*kf*256].reshape(kf,256)
        s=p.scales[block*g*512:(block+1)*g*512].reshape(g,512)
        for pair in range(2):
            q=w[:,pair*128:(pair+1)*128]
            for half,values in [(0,q&15),(1,q>>4)]:
                # Widen default even/odd ordering, then linear narrow.
                logical=np.concatenate([values[:,::2],values[:,1::2]],axis=1)
                n=block*512+pair*256+half*128
                codes[n:n+128,:kf]=logical.T
                sv=s[:,pair*256+half*128:pair*256+(half+1)*128]
                exponents[n:n+128,:g]=np.concatenate([sv[:,::2],sv[:,1::2]],axis=1).T
    if p.n>nf and kf:
        rows=p.weight[nf*kf//2:p.n*kf//2].reshape(p.n-nf,kf//2)
        codes[nf:,:kf:2]=rows&15
        codes[nf:,1:kf:2]=rows>>4
        exponents[nf:,:g]=p.scales[nf*g:p.n*g].reshape(p.n-nf,g)
    if p.k>kf:
        rows=p.weight[p.n*kf//2:].reshape(p.n,-1)
        codes[:,kf::2]=rows&15
        codes[:,kf+1::2]=(rows>>4)[:,:((p.k-kf)//2)]
        exponents[:,g]=p.scales[p.n*g:]
    return codes,exponents,np.ldexp(LUT[codes],np.repeat(exponents.astype(np.int16)-127,32,axis=1)[:,:p.k])


def original_decode(rows,scales,k):
    values=np.empty((rows.shape[0],rows.shape[1]*2),np.float64)
    values[:,::2]=LUT[rows&15]
    values[:,1::2]=LUT[rows>>4]
    return np.ldexp(values[:,:k],np.repeat(scales.astype(np.int16)-127,32,axis=1)[:,:k])


def rejects(fn):
    try: fn()
    except (ValueError,IndexError): return
    raise AssertionError('invalid input accepted')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    native_log=(out/'address-compile.log').open('w')
    command=['g++','-O2','-std=c++17','-shared','-fPIC',str(ROOT/'benchmarks/mxfp4_compact/address_probe.cpp'),'-o',str(out/'address_probe.so')]
    native_log.write(json.dumps(command)+'\n');native_log.flush()
    subprocess.run(command,stdout=native_log,stderr=subprocess.STDOUT,check=True);native_log.close()
    address=ctypes.CDLL(str(out/'address_probe.so')).restore_compact
    array_arg=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    address.argtypes=[array_arg,array_arg,ctypes.c_uint,ctypes.c_uint,array_arg,array_arg]
    address.restype=None
    rng=np.random.default_rng(27092703)
    shapes=[(n,k) for n in [1,2,63,64,127,128,129,255,256,257,511,512,513,1023,1024,1025]
                    for k in [1,2,3,31,32,33,63,64,65,256,257]]
    results=[];fixtures={};read_intervals={};native_compatible=0;total_elements=0
    for case,(n,k) in enumerate(shapes):
        rows=rng.integers(0,256,(n,(k+1)//2),dtype=np.uint8)
        scales=rng.integers(0,255,(n,(k+31)//32),dtype=np.uint8)
        if n%512==0 and k%32==0:
            scales=rng.integers(2,253,scales.shape,dtype=np.uint8)
        p=prepare(rows,scales,logical_k=k)
        r,s=unpack(p)
        assert np.array_equal(r,rows) and np.array_equal(s,scales)
        crows=np.empty_like(rows);cscales=np.empty_like(scales)
        address(p.weight,p.scales,n,k,crows,cscales)
        assert np.array_equal(crows,rows) and np.array_equal(cscales,scales)
        restored=from_buffers(p.weight,p.scales,p.metadata())
        assert np.shares_memory(restored.weight,p.weight)
        codes,es,model=schedule_decode(p);expected=original_decode(rows,scales,k)
        assert np.array_equal(es,scales)
        assert np.array_equal(model.view(np.uint64),expected.view(np.uint64))
        ledger,intervals=audit_intervals(p)
        checkpoint=rows.nbytes+scales.nbytes
        assert checkpoint==p.payload_bytes
        if p.aligned_native_v2:
            w2,s2=pack_v2(rows,scales,k)
            assert np.array_equal(w2.ravel(),p.weight) and np.array_equal(s2.ravel(),p.scales)
            v2=p.as_native_v2();assert np.shares_memory(v2.weight,p.weight) and np.shares_memory(v2.scales,p.scales)
            native_compatible+=1
        else: rejects(p.as_native_v2)
        rejects(lambda: p.code_address(n,0));rejects(lambda:p.scale_address(0,(k+31)//32))
        tag=f'n{n}_k{k}'
        # Preserve complete generated byte inputs/output storage for every audited shape.
        fixtures[tag+'_rows']=rows;fixtures[tag+'_scales']=scales
        fixtures[tag+'_compact_weight']=p.weight;fixtures[tag+'_compact_scales']=p.scales
        if (n,k) in [(1,1),(255,33),(512,32),(513,257),(1025,65)]:
            fixtures[tag+'_decoded_fp64']=model
            read_intervals[tag]=intervals
        results.append({'n':n,'k':k,'checkpoint_bytes':checkpoint,'compact_bytes':p.payload_bytes,
                        'padded_v2_bytes':((n+511)//512)*((k+31)//32)*32*256+((n+511)//512)*((k+31)//32)*512,
                        'fast_decode_eligible':p.bf16_fast_decode_eligible,'ledger':ledger,
                        'segments':[asdict(s) for s in p.segments()]})
        total_elements+=n*k
    # Every finite E8M0 code paired with each of the16 nibble codes, including signed zero.
    rows=np.tile(np.arange(8,dtype=np.uint8)*2|((np.arange(8,dtype=np.uint8)*2+1)<<4),(255,1))
    scales=np.arange(255,dtype=np.uint8)[:,None]
    p=prepare(rows,scales,logical_k=16)
    r,s=unpack(p);assert np.array_equal(r,rows) and np.array_equal(s,scales)
    _,_,model=schedule_decode(p);expected=original_decode(rows,scales,16)
    assert np.array_equal(model.view(np.uint64),expected.view(np.uint64))
    assert not p.bf16_fast_decode_eligible
    fixtures['all_finite_rows']=rows;fixtures['all_finite_scales']=scales
    fixtures['all_finite_compact_weight']=p.weight;fixtures['all_finite_compact_scales']=p.scales
    fixtures['all_finite_decoded_fp64']=model
    # Eligibility boundary and BF16 integer exponent decoding, including all q/s combinations.
    bits=LUT.astype(np.float32).view(np.uint32)>>16
    fast_scales=np.arange(2,253,dtype=np.uint16)[:,None]
    mag=bits[None,:]&32767
    adjusted=(mag+((fast_scales.astype(np.int32)-127)<<7))&32767
    adjusted=np.where(mag==0,0,adjusted)|(bits[None,:]&32768)
    fast_decoded=(adjusted.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    fast_expected=np.ldexp(LUT[None,:],fast_scales.astype(np.int32)-127)
    assert np.array_equal(fast_decoded.view(np.uint64),fast_expected.view(np.uint64))
    rejects(lambda:prepare(rows,np.full_like(scales,255),logical_k=16))
    rejects(lambda:from_buffers(p.weight,p.scales,p.metadata()[:-1]))
    bad=bytearray(p.metadata());bad[-4]=1
    rejects(lambda:from_buffers(p.weight,p.scales,bad))
    rejects(lambda:from_buffers(p.weight[:-1],p.scales,p.metadata()))
    rejects(lambda:PreparedCompact(np.empty(0,np.uint8),np.empty(0,np.uint8),2147483648,1).validate())
    for n,k in [(True,1),(1,True),(1.,1),(1,1.),(2147483647,3),(1,2147483648)]:
        rejects(lambda:PreparedCompact(np.empty(0,np.uint8),np.empty(0,np.uint8),n,k).validate())
    # End-to-end CPU FP64 oracle through decoded compact weights. Includes M reuse,
    # bias, zero, signed zeros and cancellation; this is not a native pass claim.
    oracle=[]
    for index,(m,n,k,mode) in enumerate([(1,513,257,'random'),(2,129,33,'zero'),
            (8,513,257,'cancel'),(16,512,64,'random'),(64,65,31,'random'),
            (128,17,65,'cancel'),(257,9,33,'random'),(513,3,257,'cancel')]):
        rows=rng.integers(0,256,(n,(k+1)//2),dtype=np.uint8)
        scales=rng.integers(120,135,(n,(k+31)//32),dtype=np.uint8)
        activation=rng.integers(-64,65,(m,k)).astype(np.float32)/32
        bias=rng.integers(-64,65,n).astype(np.float32)/16
        if mode=='zero': activation.fill(0)
        if mode=='cancel':
            # Equal adjacent weights and opposite exact BF16 activation values.
            rows[:,:k//2]=(rows[:,:k//2]&15)*17
            activation[:,1::2]=-activation[:,:k-1:2]
            activation[:,-1]=0
        p=prepare(rows,scales,logical_k=k);_,_,decoded=schedule_decode(p)
        expected=activation.astype(np.float64)@original_decode(rows,scales,k).T+bias
        actual=activation.astype(np.float64)@decoded.T+bias
        assert np.array_equal(actual,expected)
        if mode in ('zero','cancel'): assert np.array_equal(actual,np.broadcast_to(bias,actual.shape))
        tag=f'oracle{index}'
        fixtures.update({tag+'_rows':rows,tag+'_scales':scales,tag+'_a_f32':activation,
                         tag+'_bias_f32':bias,tag+'_compact_weight':p.weight,tag+'_compact_scales':p.scales,
                         tag+'_output_fp64':actual})
        oracle.append({'m':m,'n':n,'k':k,'mode':mode,'outputs':m*n,'cpu_equal':True})
    np.savez_compressed(out/'fixtures.npz',**fixtures)
    (out/'read-intervals.json').write_text(json.dumps(read_intervals,separators=(',',':'))+'\n')
    result={'state':'cpu_pass_tpc_device_unvalidated','seed':27092703,'shapes':len(shapes),
            'logical_weight_elements_checked':total_elements+255*16,'all_finite_scale_codes':255,
            'all_nibble_codes':16,'aligned_v2_bitwise_compatible_cases':native_compatible,'c_address_header_checked':True,
            'metadata_bytes':32,'shared_byte_lut_bytes':1024,'physical_hbm_read_bytes':'unknown',
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [Path(__file__),ROOT/'benchmarks/mxfp4_compact/address_probe.cpp',ROOT/'python/gaudi_kernels/mxfp4_compact.py',ROOT/'csrc/tpc/mxfp4_compact/address.h']},
            'cases':results,'cpu_fp64_oracle_cases':oracle}
    result['artifacts']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','source_sha256','artifacts']},indent=2))

if __name__=='__main__':main()
