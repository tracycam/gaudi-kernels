"""Adapt the audited N256 kernels to one expert-wide index space.

Keep the arithmetic body identical. Dynamic expert IDs address one original-
width weight owner; expert activations and FP32 outputs have explicit axes.
Generated source is retained in every build and covered by build hashes.
"""
from pathlib import Path


def replace_once(s,old,new):
    assert s.count(old)==1,(old,s.count(old))
    return s.replace(old,new)


def generate(root,out,activation='vector',lookup_x2=False):
    s=(root/'benchmarks/mxfp4_smallm/reuse.c').read_text()
    s=replace_once(s,'"../../csrc/tpc/mxfp4_linear/gemv.c"',str(root/'csrc/tpc/mxfp4_linear/gemv.c').__repr__().replace("'",'"'))
    s=replace_once(s,'int nb,int row,int split)','int nb,int row,int split,int expert)')
    s=replace_once(s,'int5 o={nb*256,row,split,0,0};','int5 o={nb*256,row,split,expert,0};')
    s=replace_once(s,'int5 xp##R={g*32,R,0,0,0};','int5 xp##R={g*32,R,expert,0,0};')
    s=replace_once(s,'void main(tensor packed,tensor scales,tensor table,tensor activation,tensor output)',
                   'void main(tensor packed,tensor scales,tensor table,tensor activation,tensor ids,tensor counts,tensor output)')
    s=replace_once(s,'for(int nb=begin[0];nb<end[0];++nb) {',
                   'for(int expert=begin[1];expert<end[1];++expert) {\n'
                   '        int5 ep={expert,0,0,0,0};\n'
                   '        int source=s_i32_ld_g(gen_addr(ep,ids));\n'
                   '        int count=s_i32_ld_g(gen_addr(ep,counts));\n'
                   '        for(int nb=begin[0];nb<end[0];++nb) {')
    s=s.replace('{0,g,nb,0,0}','{0,g,nb,source,0}')
    s=s.replace('{0,g*32+(Z),nb,0,0}','{0,g*32+(Z),nb,source,0}')
    s=s.replace('{0,k,nb,0,0}','{0,k,nb,source,0}')
    s=s.replace('output,nb,R,split);','output,nb,R,split,expert);')
    # Pad inactive activation rows with zero in the fixture/real grouped input.
    # Predicated MAC additionally ensures the inactive FP32 output remains zero.
    s=s.replace('Q.v1,x,a##R);','Q.v1,x,a##R,0,count>R);')
    s=s.replace('Q.v2,x,b##R);','Q.v2,x,b##R,0,count>R);')
    if activation in ('pair','inc'):
        begin=s.index('#ifdef GK_SMALLM_VECTOR\n// One K32')
        end=s.index('#if GK_SMALLM_ROWS == 1',begin)
        s=s[:begin]+'''// Exact 32-bit scalar prefetch owns two adjacent BF16 activations.
// SPU extracts high halves; remove VPU activation shuffles entirely.
#define PREFETCH(R)
#define PAIR_LOAD(R,Z) int5 px##R={g*32+(Z),R,expert,0,0}; \\
 unsigned pair##R##0=s_u32_ld_g(gen_addr(px##R,activation));px##R[0]+=2; \\
 unsigned pair##R##1=s_u32_ld_g(gen_addr(px##R,activation));
#define ROW_STEP(R,Q,Z) { unsigned bits=((Z)&2)?pair##R##1:pair##R##0; \\
 bf16 x=as_bf16((unsigned short)(bits>>(((Z)&1)*16))); \\
 a##R=v_bf16_mac_acc32_b(Q.v1,x,a##R,0,count>R);b##R=v_bf16_mac_acc32_b(Q.v2,x,b##R,0,count>R); }
#if GK_SMALLM_ROWS == 1
#define PAIR_LOAD_ALL(Z) PAIR_LOAD(0,Z)
#elif GK_SMALLM_ROWS == 2
#define PAIR_LOAD_ALL(Z) PAIR_LOAD(0,Z) PAIR_LOAD(1,Z)
#else
#define PAIR_LOAD_ALL(Z) PAIR_LOAD(0,Z) PAIR_LOAD(1,Z) PAIR_LOAD(2,Z) PAIR_LOAD(3,Z)
#endif
'''+s[end:]
        s=replace_once(s,'#define STEP4(Z) { int5 p=', '#define STEP4(Z) { PAIR_LOAD_ALL(Z) int5 p=')
        if activation=='inc':
            start=s.index('#define PAIR_LOAD(R,Z)');end=s.index('#define ROW_STEP',start)
            s=s[:start]+'''#define PAIR_LOAD(R,Z) unsigned pair##R##0=s_u32_ld_g_inc(&ap##R,SW_INC_4); \\
 unsigned pair##R##1=s_u32_ld_g_inc(&ap##R,SW_INC_4);
'''+s[end:]
            setup='''#define INIT_AP(R) int5 ax##R={split*chunk*32,R,expert,0,0}; \\
 __global unsigned* ap##R=(__global unsigned*)gen_addr(ax##R,activation);
        INIT_AP(0)
#if GK_SMALLM_ROWS >= 2
        INIT_AP(1)
#endif
#if GK_SMALLM_ROWS == 4
        INIT_AP(2) INIT_AP(3)
#endif
'''
            s=replace_once(s,'        float128 a0={0}',setup+'        float128 a0={0}')
    # Keep a four-K window available for vector activations too. The old
    # unconditional second STEP4 would otherwise overlap adjacent windows.
    s=replace_once(s,'STEP4(window) STEP4(window+4)',
                   'STEP4(window)\n#if GK_SMALLM_STRAIGHT >= 8\n                STEP4(window+4)\n#endif')
    s=replace_once(s,'#if GK_SMALLM_STRAIGHT == 32', '''#if GK_SMALLM_STRAIGHT == 2
            // Limit simultaneous decoded weights and activation broadcasts.
            // Keep the loop explicit: this variant measures pressure vs ILP.
            #pragma loop_unroll(1)
            for(int z=0;z<32;z+=2) {
                int5 p={0,g*32+z,nb,source,0};
                ushort128 i0=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1];
                ushort128 i1=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);
                bfloat256 q0=DECODE(i0),q1=DECODE(i1);
                SCALE_Q(q0) SCALE_Q(q1)
                ALL_ROWS(q0,z) ALL_ROWS(q1,z+1)
            }
#elif GK_SMALLM_STRAIGHT == 32''')
    s=s.rstrip()+'\n}\n'
    if lookup_x2:s=s.replace('SW_LUT_PTR','(SW_LUT_PTR|SW_X2)')
    (out/'grouped.c').write_text(s)

    s=(root/'csrc/tpc/mxfp4_linear/decode.c').read_text()
    s=replace_once(s,'void main(tensor packed,tensor scales,tensor table,tensor output,int block_offset)',
                   'void main(tensor packed,tensor scales,tensor table,tensor ids,tensor output)')
    s=replace_once(s,'for(int nb=begin[0];nb<end[0];++nb) {',
                   'for(int expert=begin[2];expert<end[2];++expert) {\n'
                   '      int5 ep={expert,0,0,0,0};int source=s_i32_ld_g(gen_addr(ep,ids));\n'
                   '      for(int nb=begin[0];nb<end[0];++nb) {')
    s=s.replace('nb+block_offset,0,0','nb,source,0')
    s=s.replace('{nb*256,Z,0,0,0}','{nb*256,Z,expert,0,0}')
    s=s.replace('{nb*256,k,0,0,0}','{nb*256,k,expert,0,0}')
    s=s.rstrip()+'\n}\n'
    if lookup_x2:s=s.replace('SW_LUT_PTR','(SW_LUT_PTR|SW_X2)')
    (out/'decode.c').write_text(s)

    s=(root/'csrc/tpc/mxfp4_linear/reduce.c').read_text()
    s=replace_once(s,'for(int row=begin[1];',
                   'for(int expert=begin[2];expert<end[2];++expert)for(int row=begin[1];')
    s=replace_once(s,'{block*64,row,0,0,0}','{block*64,row,0,expert,0}')
    s=replace_once(s,'p[2]=0;v_f32_st_tnsr','p[2]=expert;p[3]=0;v_f32_st_tnsr')
    (out/'reduce.c').write_text(s)
