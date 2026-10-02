"""All 16 E2M1 bit patterns times 251 legal E8M0 scales, no float oracle."""
import argparse
import json
from pathlib import Path


def old(bits, scale):
    mag=bits&0x7fff
    adjusted=(mag+((scale-127)<<7))&0x7fff
    return (bits&0x8000)|(0 if mag==0 else adjusted)


def new(bits, scale):
    return bits if bits&0x7fff==0 else (bits+((scale-127)<<7))&0xffff


def exact(bits, scale):
    if bits&0x7fff==0:return bits
    exponent=((bits>>7)&255)+scale-127
    assert 1<=exponent<=254
    return (bits&0x807f)|(exponent<<7)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    # BF16 encodings of 0,.5,1,1.5,2,3,4,6 and their negative partners.
    positive=[0,0x3f00,0x3f80,0x3fc0,0x4000,0x4040,0x4080,0x40c0]
    patterns=positive+[v|0x8000 for v in positive]
    cases=[]
    for code,bits in enumerate(patterns):
        for scale in range(2,253):
            before=old(bits,scale);after=new(bits,scale);want=exact(bits,scale)
            assert before==after==want
            if bits&0x7fff:
                assert (after&0x8000)==(bits&0x8000)
                assert 1<=(after>>7)&255<=254
            else:assert after==bits
            cases.append(dict(code=code,scale=scale,bits=bits,result=after))
    excluded=[dict(code=code,scale=scale,old=old(bits,scale),new=new(bits,scale))
              for code,bits in enumerate(patterns)for scale in (0,1,253,254,255)
              if old(bits,scale)!=new(bits,scale)]
    assert excluded, 'The legal-scale precondition must not be removed.'
    result=dict(status='PASS_EXHAUSTIVE_BITS',combinations=len(cases),
                signed_zero_cases=2*251,nonzero_result_exponent_range=[1,254],
                domain='E2M1 lookup BF16 bits, E8M0 integer codes 2..252 inclusive',
                out_of_domain_counterexamples=excluded,cases=cases,
                proof='Nonzero source exponent in126..129 plus scale-127 in-125..125 '
                      'lies in1..254. Integer addition leaves mantissa and sign unchanged. '
                      'Both zero signs bypass addition. No floating operation or FTZ.' )
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ('cases','out_of_domain_counterexamples')}))


if __name__=='__main__':main()
