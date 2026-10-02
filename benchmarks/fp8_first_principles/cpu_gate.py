"""Offline finite-domain certificate and complete CPU MAC fixtures; no device API.

The certificate concerns final FP8 encodings under explicitly modeled IEEE RNE
arithmetic. It does not certify unexecuted TPC instructions or recipe placement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from fractions import Fraction

os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np

C=np.float32(1/448)
FLOOR=np.float32(1e-10)


def decode(code):
    code=np.asarray(code,dtype=np.uint8)
    e=((code>>3)&15).astype(np.int32);m=(code&7).astype(np.float64)
    return np.where(code&128,-1.,1.)*np.where(e==0,m/512,np.ldexp(1+m/8,e-7))


LEVELS=decode(np.arange(127,dtype=np.uint8))


def ocp_code(value):
    q=np.minimum(np.abs(np.asarray(value,dtype=np.float64)),448.)
    hi=np.minimum(np.searchsorted(LEVELS,q,side='left'),126);lo=np.maximum(hi-1,0)
    dl=q-LEVELS[lo];dh=LEVELS[hi]-q
    code=np.where((dl<dh)|((dl==dh)&((lo&1)==0)),lo,hi).astype(np.uint8)
    return code|(np.signbit(value).astype(np.uint8)<<7)


def half_code(code):
    mag=np.asarray(code,dtype=np.uint8).astype(np.int32)&127
    half=np.where(mag>=16,mag-8,(mag>>1)+((mag&1)&((mag>>1)&1)))
    return ((code&128)|half.astype(np.uint8)).astype(np.uint8)


def native_formula(value):
    """Independent model of the emitted tiny-select + native conversion path."""
    magnitude=np.minimum(np.abs(value),np.float32(448))
    tiny=np.rint(magnitude*np.float32(512)).astype(np.float32)*np.float32(1/1024)
    half=np.where(magnitude<np.float32(1/32),tiny,magnitude*np.float32(.5))
    return ocp_code(np.copysign(half,value))


def bf16_bits(value):
    u=np.asarray(value,dtype=np.float32).view(np.uint32)
    return ((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)


def bf16_float(bits):
    return (np.asarray(bits,dtype=np.uint16).astype(np.uint32)<<np.uint32(16)).view(np.float32)


def fma(a,b,c):
    # Products of two FP32 operands fit the FP64 significand. For these
    # residuals, similarly scaled addition/cancellation is also exact in FP64.
    return (a.astype(np.float64)*b.astype(np.float64)+c.astype(np.float64)).astype(np.float32)


def corrected(value,scale,reciprocal):
    q=(value*reciprocal).astype(np.float32)
    return fma(fma(-q,scale,value),reciprocal,q)


def shifted(value,ulps):
    return (value.view(np.uint32).astype(np.int64)+ulps).astype(np.uint32).view(np.float32)


def reciprocal(value):
    result=np.float32(1/np.float64(value))
    target=1/Fraction(float(value))
    options=[shifted(result,k) for k in [-1,0,1]]
    # Independent rational check of the reciprocal's correctly rounded bit word.
    best=min(options,key=lambda r:(abs(Fraction(float(r))-target),int(r.view(np.uint32))&1))
    assert result.view(np.uint32)==best.view(np.uint32)
    return result


def build_table():
    normalized=np.concatenate([[0],np.ldexp(1+np.arange(128)[None,:]/128,
        np.arange(-24,1)[:,None]).ravel()]).astype(np.float32)
    table=[];entries=[];uncorrected_witnesses=[];pairs=0
    corrected_mismatches=0;fp64_code_mismatches=0
    for mantissa in range(128):
        maximum=np.float32(1+mantissa/128);x=normalized[normalized<=maximum]
        scale=np.float32(maximum*C);r0=reciprocal(scale)
        exact=x.astype(np.float64)/np.float64(scale)
        reference=half_code(ocp_code(exact.astype(np.float32)))
        fp64_code_mismatches+=int(np.count_nonzero(half_code(ocp_code(exact))!=reference))
        options=[]
        for step in range(-4,5):
            r=shifted(r0,step);actual=native_formula((x*r).astype(np.float32))
            options.append((int(np.count_nonzero(actual!=reference)),abs(step),step,r))
        misses,_,step,r=min(options,key=lambda v:v[:3]);assert misses==0
        direct=native_formula((x*r0).astype(np.float32));wrong=np.flatnonzero(direct!=reference)
        for i in wrong:
            uncorrected_witnesses.append({'amax_mantissa':mantissa,'amax':float(maximum),
                'x':float(x[i]),'scale_bits':hex(int(scale.view(np.uint32))),
                'nearest_reciprocal_bits':hex(int(r0.view(np.uint32))),
                'nearest_code':int(direct[i]),'reference_code':int(reference[i])})
        for model_reciprocal in [r0,r,shifted(r0,-2),shifted(r0,2)]:
            z=corrected(x,np.full_like(x,scale),np.full_like(x,model_reciprocal))
            corrected_mismatches+=int(np.count_nonzero(native_formula(z)!=reference))
        table.append(r);pairs+=x.size
        entries.append({'mantissa':mantissa,'ulp_adjustment':step,'reciprocal_bits':hex(int(r.view(np.uint32))),
            'examined_positive_inputs':int(x.size)})
    floor_scale=np.float32(FLOOR*C);table.append(reciprocal(floor_scale))
    assert corrected_mismatches==0 and fp64_code_mismatches==0
    return np.array(table,np.float32),normalized,{'normalized_pairs':int(pairs),'entries':entries,
        'nearest_reciprocal_counterexamples':uncorrected_witnesses,
        'corrected_reciprocal_model_code_mismatches':corrected_mismatches,
        'fp64_vs_rn32_division_code_mismatches':fp64_code_mismatches}


def full_exponent_gate(table,normalized):
    mantissas=(1+np.arange(128)/128).astype(np.float32)
    mask=normalized[None,:]<=mantissas[:,None]
    pairs=0;maxima=0;misses=0;formula_misses=0
    for exponent in range(-34,128):
        maximum=np.ldexp(mantissas,exponent).astype(np.float32)
        active=maximum>=FLOOR;valid=mask&active[:,None]
        x=np.ldexp(normalized,exponent).astype(np.float32)
        scale=(maximum*C).astype(np.float32)
        rb=(table[:128].view(np.uint32).astype(np.int64)-exponent*0x800000).astype(np.uint32)
        r=rb.view(np.float32)
        reference=half_code(ocp_code((x.astype(np.float64)[None,:]/scale.astype(np.float64)[:,None]).astype(np.float32)))
        z=(x[None,:]*r[:,None]).astype(np.float32)
        candidate=native_formula(z)
        misses+=int(np.count_nonzero((candidate!=reference)&valid))
        formula_misses+=int(np.count_nonzero((candidate!=half_code(ocp_code(z)))&valid))
        pairs+=int(valid.sum());maxima+=int(active.sum())
    # Below the clamp: all positive BF16 patterns, including every subnormal.
    bits=np.arange(0,0x2edc,dtype=np.uint16);x=bf16_float(bits)
    scale=np.float32(FLOOR*C)
    reference=half_code(ocp_code((x.astype(np.float64)/np.float64(scale)).astype(np.float32)))
    actual=native_formula((x*table[128]).astype(np.float32))
    clamp_misses=int(np.count_nonzero(actual!=reference))
    assert misses==formula_misses==clamp_misses==0
    return {'normal_amax_patterns':maxima,'actual_exponent_pairs':pairs,'code_mismatches':misses,
        'native_formula_vs_two_rounding_mismatches':formula_misses,'clamp_input_patterns':int(bits.size),
        'clamp_mismatches':clamp_misses,'sign_rule':'Magnitude certificate extends by explicit source sign-bit restoration.',
        'omitted_small_inputs':'For exponent delta<-24, x/scale<448*2^-23, far below the first nonzero native threshold3/1024; both encode zero.'}


def lookup_for_rows(x,table):
    raw=bf16_bits(np.max(np.abs(x),axis=1,keepdims=True))
    normal=raw>0x2edb;indices=np.where(normal,raw&127,128)
    exponents=np.where(normal,(raw.astype(np.int32)>>7)-127,0)
    bits=table[indices].view(np.uint32).astype(np.int64)-exponents*0x800000
    return bits.astype(np.uint32).view(np.float32)


def lookup_shuffle_gate(table):
    """Byte-level model of the actual shuffle/group selection, not table indexing."""
    low=table[:64].view(np.uint8).reshape(4,64)
    high=table[64:128].view(np.uint8).reshape(4,64)
    checked=0
    for mantissa in range(128):
        control=np.array([0x83828180+(mantissa&15)*0x04040404],np.uint32)
        byte_indices=control.view(np.uint8)&63
        # Each dual group contains 16 FP32 lanes. SHUFFLE high control bit
        # enables the lane; the low six bits address within that dual group.
        lo=np.tile(low[:,byte_indices],(1,16)).copy().reshape(-1).view(np.float32)
        hi=np.tile(high[:,byte_indices],(1,16)).copy().reshape(-1).view(np.float32)
        selected=(hi if mantissa>=64 else lo).reshape(4,16).copy()
        selected[np.arange(4)!=((mantissa>>4)&3)]=0
        selected=np.maximum(selected,selected[[1,0,3,2]])
        selected=np.maximum(selected,selected[[2,3,0,1]])
        assert np.all(selected.view(np.uint32)==table[mantissa].view(np.uint32))
        checked+=selected.size
    return {'entries':128,'broadcast_lanes_checked':checked,'mismatches':0,
        'scope':'Independent byte/group ISA semantics model; device execution pending.'}


def edge_gate(table):
    records=[]
    for maximum_bits in [0,1,127,128,0x2edb,0x2edc,0x3fd9,0x7f7f]:
        bits=np.unique(np.array([0,1,127,128,maximum_bits//2,maximum_bits],np.uint16))
        bits=bits[bits<=maximum_bits]
        bits=np.concatenate([bits,bits|0x8000]);x=bf16_float(bits)
        maximum=bf16_float(np.uint16(maximum_bits))
        scale=np.float32(max(maximum,FLOOR)*C)
        r=lookup_for_rows(np.array([[maximum]],np.float32),table)[0,0]
        ref=half_code(ocp_code((x.astype(np.float64)/np.float64(scale)).astype(np.float32)))
        actual=native_formula((x*r).astype(np.float32))
        assert np.array_equal(ref,actual)
        records.append({'amax_bf16_bits':hex(maximum_bits),'input_bf16_bits':[hex(int(i)) for i in bits],
            'native_codes':actual.tolist(),'mismatches':0})
    return records


def amax_gate():
    # Exhaustive ordering check, then mirror four independent vector chains
    # with explicit zero padding. Device tensor-padding behavior is pending.
    bits=np.arange(0x7f80,dtype=np.uint16)
    floats=bf16_float(bits)
    assert np.all(floats[1:]>floats[:-1])
    rng=np.random.default_rng(123);records=[]
    for k in [1,127,128,129,257,511,512,513,2048,2049,8193,32640]:
        raw=rng.integers(0,0x7f80,(8,k),dtype=np.uint16)
        raw|=rng.integers(0,2,raw.shape,dtype=np.uint16)<<15
        raw[0,:]=np.resize(np.arange(0x7f80,dtype=np.uint16),k)
        padded=np.pad(raw&0x7fff,((0,0),(0,(-k)%512)))
        chains=padded.reshape(8,-1,4,128).max(axis=1)
        reduced=chains.max(axis=(1,2))
        expected=np.max(np.abs(bf16_float(raw)),axis=1)
        assert np.array_equal(bf16_float(reduced),expected)
        records.append({'K':k,'rows':8,'mismatches':0})
    return {'finite_magnitude_encodings':int(bits.size),'four_chain_zero_padded_cases':records}


def mac_fixtures(out,table):
    records=[]
    for m,n,k,kind in [(1,129,257,'random'),(8,257,2049,'ties'),(512,1024,2048,'random'),
                       (2,255,513,'range'),(8,257,769,'cancellation'),(1,135,255,'zero')]:
        p=out/f'{kind}-{m}-{n}-{k}';p.mkdir()
        rng=np.random.default_rng(13+m+n+k)
        x=bf16_float(bf16_bits(rng.normal(0,.3,(m,k))))
        w=rng.integers(0,127,(n,k),dtype=np.uint8)|(rng.integers(0,2,(n,k),dtype=np.uint8)<<7)
        ws=np.exp2(rng.uniform(-12,-4,n)).astype(np.float32);bias=rng.normal(0,.01,n).astype(np.float32)
        if kind=='ties':
            values=np.concatenate([LEVELS,(LEVELS[:-1]+LEVELS[1:])*.5])/64
            u=bf16_bits(values)
            v=bf16_float(np.concatenate([np.maximum(u.astype(np.int32)-1,0).astype(np.uint16),u,u+1]))
            v=np.clip(np.concatenate([v,-v]),-7,7)
            for row in range(m):x[row]=np.resize(np.roll(v,row*137),k)
            x[:,-1]=7;bias[:]=0
        if kind=='range':
            x[:]=bf16_float(bf16_bits(np.resize(np.array([0,1e-38,-1e-38,1e-12,-1e-12,1e-10,10,-10],np.float32),x.shape)))
            ws=np.exp2(np.linspace(-20,0,n)).astype(np.float32)
        if kind=='cancellation':
            x[:,1::2]=-x[:,0:k-1:2];w[:,1::2]=w[:,0:k-1:2];x[:,-1]=0
        if kind=='zero':x[:]=0;bias[:]=0
        scale=(np.maximum(np.max(np.abs(x),axis=1,keepdims=True),FLOOR)*C).astype(np.float32)
        q_exact=ocp_code(x.astype(np.float64)/scale.astype(np.float64))
        q_reference=half_code(q_exact)
        q_candidate=native_formula((x*lookup_for_rows(x,table)).astype(np.float32))
        q_mismatches=int(np.count_nonzero(q_candidate!=q_reference));assert q_mismatches==0
        adapt=np.any((w&127)>119,axis=1,keepdims=True);wn=np.where(adapt,half_code(w),w)
        prepared_scale=np.where(adapt[:,0],ws*2,ws).astype(np.float32)
        original=x.astype(np.float64)@(decode(w)*ws[:,None]).T+bias
        gpu=(decode(q_exact)*scale)@(decode(w)*ws[:,None]).T+bias
        baseline=(decode(q_reference)*(scale*2))@(decode(wn)*prepared_scale[:,None]).T+bias
        candidate=(decode(q_candidate)*(scale*2))@(decode(wn)*prepared_scale[:,None]).T+bias
        baseline_bf16=bf16_bits(baseline);candidate_bf16=bf16_bits(candidate)
        assert np.array_equal(baseline_bf16,candidate_bf16)
        values={'activation_bf16':bf16_bits(x),'weight_original':w,'weight_prepared':wn,'weight_scales':ws,
            'bias':bias,'activation_scale':scale,'reference_codes':q_reference,'candidate_codes':q_candidate,
            'original_fp64':original,'gpu_style_fp64':gpu,'adapted_fp64':baseline,'candidate_fp64':candidate,
            'candidate_bf16':candidate_bf16}
        for name,value in values.items():np.save(p/(name+'.npy'),value)
        actual=bf16_float(candidate_bf16).astype(np.float64)
        record={'M':m,'N':n,'K':k,'kind':kind,'activation_codes_checked':int(x.size),'code_mismatches':q_mismatches,
            'complete_outputs':m*n,'changed_bf16_outputs':0,
            'relative_fp64':float(np.linalg.norm(actual-original)/max(np.linalg.norm(original),1e-30)),
            'relative_gpu_style_fp64':float(np.linalg.norm(actual-gpu)/max(np.linalg.norm(gpu),1e-30)),
            'scope':'CPU FP64 MAC and BF16 final rounding; not device MME accumulation/timing',
            'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.glob('*.npy')}}
        (p/'result.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    table,normalized,certificate=build_table();certificate.update(full_exponent_gate(table,normalized))
    certificate['lookup_shuffle_gate']=lookup_shuffle_gate(table)
    certificate['edge_gate']=edge_gate(table)
    certificate['amax_gate']=amax_gate()
    certificate['baseline_scope']='Reference is RN32(x / contractual FP32 scale), then OCP RNE and native half RNE. Corrected-reciprocal comparisons use modeled RN32 reciprocal, selected LUT reciprocal, and RN32 reciprocal +/-2 ULP. They do not emulate the device reciprocal SFU polynomial or certify its error bound.'
    table.tofile(out/'reciprocal_table_f32.bin')
    certificate.update(table_bytes=int(table.nbytes),table_sha256=hashlib.sha256(table.tobytes()).hexdigest(),
        scale_multiplier_bits=hex(int(C.view(np.uint32))),floor_bits=hex(int(FLOOR.view(np.uint32))),
        table_words=[hex(int(v)) for v in table.view(np.uint32)],
        runtime_verified=False,scope='Finite BF16 CPU arithmetic certificate, not an HPU execution certificate')
    (out/'certificate.json').write_text(json.dumps(certificate,indent=2)+'\n')
    records=mac_fixtures(out,table)
    (out/'mac-results.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({k:v for k,v in certificate.items() if k not in ['entries','table_words']},indent=2))
    print('Complete CPU MAC outputs checked:',sum(r['complete_outputs'] for r in records))


if __name__=='__main__':main()
