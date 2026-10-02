"""Original-layout AV edit of actual quad ELF disassembly, preserving other packets."""
import argparse,hashlib,json,re
from pathlib import Path

def parse(path):
    out=[];active=False
    for raw in path.read_text().splitlines():
        label=re.fullmatch(r'[0-9a-f]+ (main|\.LBB\w+):',raw.strip())
        if label:out.append(label[1]+':');active=True
        elif active:
            hit=re.match(r'\s*[0-9a-f]+:\s+(.*)',raw)
            if hit:out.append(hit[1].split('//',1)[0].strip())
            elif raw.startswith(('Disassembly of section','.section','TPC compiler')):break
    return out

def patch(original):
    begin=next(i for i,s in enumerate(original)if re.fullmatch(r'loop S13, 0, -1, >, \.LBB0_27, !SP2',s))
    end=original.index('.LBB0_27:')
    nop='nop; nop; nop; nop'
    replacement=['nop; jmpr .LBB0_27, SP2; nop; nop',nop,'.LBB0_19:',
        # Score index S10, value slot S9, remaining token count S13.
        'nop; and.i32 S11, S10, 0xf; mov V11, V4; nop',
        'nop; shr.i32 S12, S10, 0x4; nop; nop',
        'nop; cmp_less.i32 SP2, S10, 0x40; nop; nop',
        'nop; nop; nop; nop',
        nop, # SHR->AND scalar result requires four issue packets.
        'nop; and.i32 S12, S12, 0x3; mov V11, V5, !SP2; nop',
        'nop; sub.i32 neg S15, S11, 0x10; nop; nop',
        'nop; mul.i32 S16, S11, 0x4040404; nop; nop',
        'nop; nop; nop; nop',
        'nop; cmp_eq.i32 SP3, S12, 0x0; nop; nop',
        'nop; min.i32 S15, S15, S13; nop; nop',
        'nop; cmp_eq.i32 SP4, S12, 0x1; nop; nop',
        'nop; cmp_eq.i32 SP5, S12, 0x2; mov_dg.all sdg0=0 sdg1=0 sdg2=0 sdg3=0 weg0=3 weg1=3 weg2=3 weg3=3 V20, V11, 0xffffffff, SP3; nop',
        'nop; cmp_eq.i32 SP6, S12, 0x3; nop; nop',
        'nop; add.i32 S16, S16, 0x83828180; nop; nop',
        'nop; nop; mov_dg.all sdg0=1 sdg1=1 sdg2=1 sdg3=1 weg0=3 weg1=3 weg2=3 weg3=3 V20, V11, 0xffffffff, SP4; nop',
        'nop; nop; mov_dg.all sdg0=2 sdg1=2 sdg2=2 sdg3=2 weg0=3 weg1=3 weg2=3 weg3=3 V20, V11, 0xffffffff, SP5; nop',
        'nop; nop; mov_dg.all sdg0=3 sdg1=3 sdg2=3 sdg3=3 weg0=3 weg1=3 weg2=3 weg3=3 V20, V11, 0xffffffff, SP6; nop',
        'mov.i32 V21, S16; nop; nop; nop',nop,nop,nop,nop,
        'loop S15, 0, -1, >, .LBB0_24',nop,'.LBB0_23:',
        # One probability and one original-layout V load, chronological D2 FMA.
        'mov b11111 I6, I4; add.i32 S10, S10, 0x1; shuffle.u8 V10, V20, V21; nop',
        'set_indx I6, b00100, S9; add.i32 S9, S9, 0x1; nop; nop',
        'ld_tnsr V11, 0x2, I6; nop; add.i32 V21, V21, 0x4040404; nop',nop,nop,nop,
        'nop; nop; convert.bf16 all_lanes target_type=fp32 rhne D12, V11; nop',nop,nop,nop,
        'nop; nop; mac.f32 x2 D2, D12, V10, V10; nop',
        '.LBB0_24:', 'nop; sub.i32 S13, S13, S15; nop; nop',nop,nop,nop,
        '.LBB0_25:', 'nop; cmp_grt.i32 SP3, S13, 0x0; nop; nop',nop,nop,
        '.LBB0_26:', 'nop; jmpr .LBB0_19, SP3; nop; nop',nop]
    result=original[:begin]+replacement+original[end:]
    assert {s for s in result if s.endswith(':')}=={s for s in original if s.endswith(':')}
    assert result[:begin]==original[:begin] and result[-len(original[end:]):]==original[end:]
    return result,{'replaced_packets':sum(not s.endswith(':')for s in original[begin:end]),
                   'replacement_static_packets':sum(not s.endswith(':')for s in replacement),
                   'inner_packets_per_token':11,'group_width_max':16,
                   'QK_and_softmax_and_reciprocal_original_packets_preserved':True,
                   'AV_order':'same chronological mac.f32 x2 D2 recurrence, no independent partial sums',
                   'V_liveness':'V2/V3 accumulator, V4/V5 probabilities, V9 denominator; V10 probability,V11 BF16 load,V12/V13 conversion,V20 group,V21 byte controls; no new memory scratch'}

def debug(original):
    assert not any(re.search(r'\bI1[01]\b',s)for s in original)
    result=[];positions=[]
    for line in original:
        if re.search(r'convert.f32 all_lanes target_type=bf16 rhne V2, D2',line):
            result+=['nop; nop; nop; st_tnsr 0xa, I10, V2','nop; nop; nop; st_tnsr 0xa, I11, V3']
        positions.append(len(result));result.append(line)
        if line=='.LBB0_5:':
            result+=['set_indx I10, b00010, S32; nop; nop; nop','set_indx I10, b11101, 0x0; nop; nop; nop',
                     'set_indx I11, b00010, S32; nop; nop; nop','set_indx I11, b11101, 0x0; nop; nop; nop',
                     'set_indx I11, b00001, 0x40; nop; nop; nop']
        elif line=='.LBB0_16:':result+=['nop; nop; nop; st_tnsr 0x8, I10, V9','nop; nop; nop; st_tnsr 0x8, I11, V10']
        elif line=='.LBB0_28:':result+=['nop; nop; nop; st_tnsr 0x9, I10, V2','nop; nop; nop; st_tnsr 0x9, I11, V3']
    assert [result[i]for i in positions]==original and len(result)-len(original)==11
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);old=parse(a.source);new,meta=patch(old)
    for name,lines in [('quad',old),('group',new),('quad_debug',debug(old)),('group_debug',debug(new))]:
        (a.output/(name+'.s')).write_text('.text\n.globl main\n'+'\n'.join(lines)+'\n')
    meta.update(disassembly_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),debug_existing_packets_modified=0,debug_packets_added=11)
    (a.output/'patch.json').write_text(json.dumps(meta,indent=2)+'\n')
