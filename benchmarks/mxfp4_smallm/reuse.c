// Private M=1/2/4 specialization, same N256 original-width layout as the MME
// decoder. One task owns every M row; no packed-weight reread across rows.
// Folded mode changes FP32 summation grouping, not input/output precision.
// It is experimental: unscaled block sums need an activation range guard
// before general production use. Benchmark fixtures explicitly check outputs.
#define main unused_legacy_main
#include "../../csrc/tpc/mxfp4_linear/gemv.c"
#undef main

static inline bfloat128 scale_exact(bfloat128 q,ushort128 s) {
    return v_bf16_mul_vb(q,(bfloat128)((s&255)<<7),0,q,v_bf16_cmp_neq_b(q,0));
}
static inline float128 fold(float128 part,float128 scale,float128 total) {
    total.v1=v_f32_mac_b(part.v1,scale.v1,total.v1);
    total.v2=v_f32_mac_b(part.v2,scale.v2,total.v2);return total;
}
static inline void store_row(float128 a,float128 b,tensor output,int nb,int row,int split) {
    a=linear_acc(a);b=linear_acc(b);int5 o={nb*256,row,split,0,0};
    v_f32_st_tnsr(o,output,a.v1);o[0]+=64;v_f32_st_tnsr(o,output,a.v2);
    o[0]+=64;v_f32_st_tnsr(o,output,b.v1);o[0]+=64;v_f32_st_tnsr(o,output,b.v2);
}
#ifdef GK_SMALLM_VECTOR
// One K32 activation fetch per row, replicated to all dual groups. Byte
// shuffle broadcasts an exact BF16 bit pattern; no activation quantization.
#define PREFETCH(R) int5 xp##R={g*32,R,0,0,0}; \
 uchar256 xv##R=(uchar256)v_bf16_ld_tnsr_partial_b(xp##R,activation,31,0); \
 xv##R=v_u8_mov_dual_group_all_b(xv##R,-1,0,0,0,0,MkWrA(3,3,3,3),0);
#define ROW_STEP(R,Q,Z) { uchar256 mask=(uchar256)v_u32_mov_b(0x81808180u+((unsigned)(Z)&31)*0x02020202u); \
 bfloat128 x=(bfloat128)v_u8_shuffle_b(xv##R,mask,0,0); \
 a##R=v_bf16_mac_acc32_b(Q.v1,x,a##R);b##R=v_bf16_mac_acc32_b(Q.v2,x,b##R); }
#else
#define ROW_STEP(R,Q,Z) { int5 xp={Z,R,0,0,0}; \
 bf16 x=s_bf16_ld_g(gen_addr(xp,activation)); \
 a##R=v_bf16_mac_acc32_b(Q.v1,x,a##R);b##R=v_bf16_mac_acc32_b(Q.v2,x,b##R); }
#endif
#if GK_SMALLM_ROWS == 1
#define ALL_ROWS(Q,Z) ROW_STEP(0,Q,Z)
#elif GK_SMALLM_ROWS == 2
#define ALL_ROWS(Q,Z) ROW_STEP(0,Q,Z) ROW_STEP(1,Q,Z)
#elif GK_SMALLM_ROWS == 4
#define ALL_ROWS(Q,Z) ROW_STEP(0,Q,Z) ROW_STEP(1,Q,Z) ROW_STEP(2,Q,Z) ROW_STEP(3,Q,Z)
#endif
#ifdef GK_SMALLM_FOLD_SCALE
#define SCALE_Q(Q)
#else
#define SCALE_Q(Q) Q.v1=scale_exact(Q.v1,s0);Q.v2=scale_exact(Q.v2,s1);
#endif
#define DECODE(I) v_bf16_lookup_2c(I,0,SW_LUT_PTR,(bfloat256){0})
void main(tensor packed,tensor scales,tensor table,tensor activation,tensor output) {
    set_lut_256(table);
    int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    int K=get_dim_size(activation,0),groups=(K+31)/32;
    int splits=get_dim_size(output,2),chunk=(groups+splits-1)/splits;
    for(int split=begin[2];split<end[2];++split)
    for(int nb=begin[0];nb<end[0];++nb) {
        float128 a0={0},b0={0},t0={0},u0={0};
#if GK_SMALLM_ROWS >= 2
        float128 a1={0},b1={0},t1={0},u1={0};
#endif
#if GK_SMALLM_ROWS == 4
        float128 a2={0},b2={0},t2={0},u2={0},a3={0},b3={0},t3={0},u3={0};
#endif
        for(int g=split*chunk;g<(split+1)*chunk&&g<groups;++g) {
#ifndef GK_SMALLM_FOLD_SCALE
            int5 sp={0,g,nb,0,0};
            ushort128 s0=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);sp[0]=128;
            ushort128 s1=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);
#endif
            int k=g*32;
#ifdef GK_SMALLM_VECTOR
            PREFETCH(0)
#if GK_SMALLM_ROWS >= 2
            PREFETCH(1)
#endif
#if GK_SMALLM_ROWS == 4
            PREFETCH(2) PREFETCH(3)
#endif
#endif
#if GK_SMALLM_UNROLL == 4
#ifdef GK_SMALLM_STRAIGHT
            // The actual compiler left the K4 loop despite loop_unroll(8).
            // Eight explicit bodies expose constant shuffle masks and remove
            // the per-K4 loop/control chain. Keep lexical register lifetimes.
#define STEP4(Z) { int5 p={0,g*32+(Z),nb,0,0}; \
 ushort128 i0=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1]; \
 ushort128 i1=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1]; \
 ushort128 i2=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1]; \
 ushort128 i3=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16); \
 bfloat256 q0=DECODE(i0),q1=DECODE(i1),q2=DECODE(i2),q3=DECODE(i3); \
 SCALE_Q(q0) SCALE_Q(q1) SCALE_Q(q2) SCALE_Q(q3) \
 ALL_ROWS(q0,(Z)) ALL_ROWS(q1,(Z)+1) ALL_ROWS(q2,(Z)+2) ALL_ROWS(q3,(Z)+3) }
#if GK_SMALLM_STRAIGHT == 32
            STEP4(0) STEP4(4) STEP4(8) STEP4(12) STEP4(16) STEP4(20) STEP4(24) STEP4(28)
#else
            for(int window=0;window<32;window+=GK_SMALLM_STRAIGHT){
                STEP4(window) STEP4(window+4)
#if GK_SMALLM_STRAIGHT == 16
                STEP4(window+8) STEP4(window+12)
#endif
            }
#endif
#undef STEP4
#else
#ifdef GK_SMALLM_VECTOR
            #pragma loop_unroll(8)
            for(int z=0;z<32;z+=4) { k=g*32+z;
#else
            for(;k+3<(g+1)*32&&k+3<K;k+=4) {
#endif
                int5 p={0,k,nb,0,0};
                ushort128 i0=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1];
                ushort128 i1=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1];
                ushort128 i2=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);++p[1];
                ushort128 i3=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);
                bfloat256 q0=DECODE(i0),q1=DECODE(i1),q2=DECODE(i2),q3=DECODE(i3);
                SCALE_Q(q0) SCALE_Q(q1) SCALE_Q(q2) SCALE_Q(q3)
                ALL_ROWS(q0,k) ALL_ROWS(q1,k+1) ALL_ROWS(q2,k+2) ALL_ROWS(q3,k+3)
            }
#endif
#endif
#ifndef GK_SMALLM_VECTOR
            for(;k<(g+1)*32&&k<K;++k) {
                int5 p={0,k,nb,0,0};
                ushort128 index=(ushort128)v_u8_ld_tnsr_b(p,packed,SW_UNPACK|SW_UNPCK_8_TO_16);
                bfloat256 q=DECODE(index);SCALE_Q(q) ALL_ROWS(q,k)
            }
#endif
#ifdef GK_SMALLM_FOLD_SCALE
            // Scale is only used after the K32 block. Delay both loads and
            // conversion rather than retaining scale vectors through MACs.
            int5 sp={0,g,nb,0,0};
            ushort128 s0=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);sp[0]=128;
            ushort128 s1=(ushort128)v_u8_ld_tnsr_b(sp,scales,SW_UNPACK|SW_UNPCK_8_TO_16);
            float128 f0=v_convert_bf16_to_f32_all_b((bfloat128)((s0&255)<<7));
            float128 f1=v_convert_bf16_to_f32_all_b((bfloat128)((s1&255)<<7));
#define FOLD_ROW(R) t##R=fold(a##R,f0,t##R);u##R=fold(b##R,f1,u##R);a##R=(float128){0};b##R=(float128){0};
            FOLD_ROW(0)
#if GK_SMALLM_ROWS >= 2
            FOLD_ROW(1)
#endif
#if GK_SMALLM_ROWS == 4
            FOLD_ROW(2) FOLD_ROW(3)
#endif
#endif
        }
#ifdef GK_SMALLM_FOLD_SCALE
#define STORE(R) store_row(t##R,u##R,output,nb,R,split);
#else
#define STORE(R) store_row(a##R,b##R,output,nb,R,split);
#endif
        STORE(0)
#if GK_SMALLM_ROWS >= 2
        STORE(1)
#endif
#if GK_SMALLM_ROWS == 4
        STORE(2) STORE(3)
#endif
    }
}
