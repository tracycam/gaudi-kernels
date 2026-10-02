// Experimental FP32 scale/FMA chains. Original partials, scales and BF16 output.
// One row per task: chain8 uses 16 accumulator vectors, rather than multiplying
// the live set by four rows and forcing spills. The M>1 cost must be measured.
#ifndef CHAINS
#define CHAINS 4
#endif
#if CHAINS != 4 && CHAINS != 8
#error supported chain counts are 4 and 8
#endif
void main(tensor partial,tensor activation_scales,tensor weight_scales,tensor bias,tensor output) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int groups=get_dim_size(partial,2);
 for(int row=begin[1];row<end[1];++row) {
  for(int nb=begin[0];nb<end[0];++nb) {
   int col=nb*128;
   float64 lo0=0,hi0=0,lo1=0,hi1=0,lo2=0,hi2=0,lo3=0,hi3=0;
#if CHAINS == 8
   float64 lo4=0,hi4=0,lo5=0,hi5=0,lo6=0,hi6=0,lo7=0,hi7=0;
#endif
   for(int g=0;g<groups;g+=CHAINS) {
#define STEP(I) { \
    bool live=g+I<groups; \
    int5 wp={g+I,nb,0,0,0},ap={0,row,g+I,0,0},pp={col,row,g+I,0,0}; \
    float64 ws=v_f32_ld_g(gen_addr(wp,weight_scales),0,0,live); \
    float64 as=v_f32_ld_g(gen_addr(ap,activation_scales),0,0,live); \
    float64 factor=ws*as; \
    float64 lo=v_f32_ld_tnsr_b(pp,partial,0,0,live); pp[0]+=64; \
    float64 hi=v_f32_ld_tnsr_b(pp,partial,0,0,live); \
    lo##I=v_f32_mac_b(lo,factor,lo##I); hi##I=v_f32_mac_b(hi,factor,hi##I); \
   }
    STEP(0) STEP(1) STEP(2) STEP(3)
#if CHAINS == 8
    STEP(4) STEP(5) STEP(6) STEP(7)
#endif
#undef STEP
   }
   float64 lo=(lo0+lo1)+(lo2+lo3),hi=(hi0+hi1)+(hi2+hi3);
#if CHAINS == 8
   lo=lo+((lo4+lo5)+(lo6+lo7));hi=hi+((hi4+hi5)+(hi6+hi7));
#endif
   int5 b={col,0,0,0,0};float64 blo=v_f32_ld_tnsr_b(b,bias);b[0]+=64;float64 bhi=v_f32_ld_tnsr_b(b,bias);
   float128 pair={lo+blo,hi+bhi};int5 dst={col,row,0,0,0};
   v_bf16_st_tnsr(dst,output,convert_float128_to_bfloat128(pair,SW_LINEAR|SW_RHNE));
  }
 }
}
