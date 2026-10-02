// M1 grouped expert grid. One full 256-byte load supplies 512 weights.
// N512 is the historical row-pair layout, a lossless interleave of N256 panels.
#define main unused_legacy_main
#include "../../csrc/tpc/mxfp4_linear/gemv.c"
#undef main
#define STEP(Z) { int5 p={0,g*32+(Z),nb,source,0}; \
 uchar256 raw=v_u8_ld_tnsr_b(p,packed); \
 ushort256 index=convert_uchar256_to_ushort256(raw,SW_LINEAR); \
 bfloat256 q0=v_bf16_lookup_2c(index.v1,0,SW_LUT_PTR,(bfloat256){0}); \
 bfloat256 q1=v_bf16_lookup_2c(index.v2,0,SW_LUT_PTR,(bfloat256){0}); \
 uchar256 mask=(uchar256)v_u32_mov_b(0x81808180u+((unsigned)(Z)&31)*0x02020202u); \
 bfloat128 x=(bfloat128)v_u8_shuffle_b(xv,mask,0,0); \
 p0=v_bf16_mac_acc32_b(q0.v1,x,p0);p1=v_bf16_mac_acc32_b(q0.v2,x,p1); \
 p2=v_bf16_mac_acc32_b(q1.v1,x,p2);p3=v_bf16_mac_acc32_b(q1.v2,x,p3); }
#define FOLD(R,S) { float128 f=v_convert_bf16_to_f32_all_b((bfloat128)(((S)&255)<<7)); \
 t##R.v1=v_f32_mac_b(p##R.v1,f.v1,t##R.v1);t##R.v2=v_f32_mac_b(p##R.v2,f.v2,t##R.v2); }
#define STORE(R) { float128 linear=linear_acc(t##R);int5 dst={nb*512+(R)*128,0,split,expert,0}; \
 v_f32_st_tnsr(dst,output,linear.v1);dst[0]+=64;v_f32_st_tnsr(dst,output,linear.v2); }
void main(tensor packed,tensor scales,tensor table,tensor activation,tensor ids,tensor counts,tensor output){
 set_lut_256(table);int5 b=get_index_space_offset(),end=b+get_index_space_size();
 int K=get_dim_size(activation,0),splits=get_dim_size(output,2),chunk=(K/32+splits-1)/splits;
 for(int split=b[2];split<end[2];++split)for(int expert=b[1];expert<end[1];++expert){
  int5 ep={expert,0,0,0,0};int source=s_i32_ld_g(gen_addr(ep,ids));
  for(int nb=b[0];nb<end[0];++nb){
   float128 t0={0},t1={0},t2={0},t3={0};
   for(int g=split*chunk;g<(split+1)*chunk&&g<K/32;++g){
    float128 p0={0},p1={0},p2={0},p3={0};
    int5 xp={g*32,0,expert,0,0};uchar256 xv=(uchar256)v_bf16_ld_tnsr_partial_b(xp,activation,31,0);
    xv=v_u8_mov_dual_group_all_b(xv,-1,0,0,0,0,MkWrA(3,3,3,3),0);
#if GK_N512_WINDOW == 32
    STEP(0) STEP(1) STEP(2) STEP(3) STEP(4) STEP(5) STEP(6) STEP(7)
    STEP(8) STEP(9) STEP(10) STEP(11) STEP(12) STEP(13) STEP(14) STEP(15)
    STEP(16) STEP(17) STEP(18) STEP(19) STEP(20) STEP(21) STEP(22) STEP(23)
    STEP(24) STEP(25) STEP(26) STEP(27) STEP(28) STEP(29) STEP(30) STEP(31)
#else
    for(int z=0;z<32;z+=GK_N512_WINDOW){
     STEP(z) STEP(z+1)
#if GK_N512_WINDOW >= 4
     STEP(z+2) STEP(z+3)
#endif
#if GK_N512_WINDOW >= 8
     STEP(z+4) STEP(z+5) STEP(z+6) STEP(z+7)
#endif
    }
#endif
    int5 sp={0,g,nb,source,0};ushort256 s0=convert_uchar256_to_ushort256(v_u8_ld_tnsr_b(sp,scales),SW_LINEAR);sp[0]=256;
    ushort256 s1=convert_uchar256_to_ushort256(v_u8_ld_tnsr_b(sp,scales),SW_LINEAR);
    FOLD(0,s0.v1) FOLD(1,s0.v2) FOLD(2,s1.v1) FOLD(3,s1.v2)
   }
   STORE(0) STORE(1) STORE(2) STORE(3)
  }
 }
}
