// Compact original block scales, broadcast directly into vector registers.
// The compiler must keep [G,M,N] partials in SRAM; audit separately.
#ifndef ROWS
#define ROWS 1
#endif
void main(tensor partial,tensor activation_scales,tensor weight_scales,tensor bias,tensor output) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int groups=get_dim_size(partial,2),rows=get_dim_size(output,1);
 for(int rb=begin[1];rb<end[1];++rb) {
  int row=rb*ROWS;
  for(int nb=begin[0];nb<end[0];++nb) {
   int col=nb*128;
   float64 lo0=0,hi0=0;
#if ROWS == 4
   float64 lo1=0,hi1=0,lo2=0,hi2=0,lo3=0,hi3=0;
#endif
   #pragma loop_unroll(4)
   for(int g=0;g<groups;++g) {
    int5 wp={g,nb,0,0,0};
    float64 ws=v_f32_ld_g(gen_addr(wp,weight_scales));
#define STEP(R) { \
    bool live=row+R<rows; \
    int5 ap={0,row+R,g,0,0},pp={col,row+R,g,0,0}; \
    float64 as=v_f32_ld_g(gen_addr(ap,activation_scales),0,0,live); \
    float64 factor=ws*as; \
    float64 lo=v_f32_ld_tnsr_b(pp,partial,0,0,live); pp[0]+=64; \
    float64 hi=v_f32_ld_tnsr_b(pp,partial,0,0,live); \
    lo##R=v_f32_mac_b(lo,factor,lo##R); hi##R=v_f32_mac_b(hi,factor,hi##R); \
   }
    STEP(0)
#if ROWS == 4
    STEP(1) STEP(2) STEP(3)
#endif
#undef STEP
   }
#if OUTPUT_BF16
#define STORE(R) {int5 b={col,0,0,0,0};float64 blo=v_f32_ld_tnsr_b(b,bias);b[0]+=64;float64 bhi=v_f32_ld_tnsr_b(b,bias);int5 p={col,row+R,0,0,0};float128 pair={lo##R+blo,hi##R+bhi};v_bf16_st_tnsr(p,output,convert_float128_to_bfloat128(pair,SW_LINEAR|SW_RHNE),0,row+R<rows);}
#else
#define STORE(R) {int5 p={col,row+R,0,0,0};v_f32_st_tnsr(p,output,lo##R,0,row+R<rows);p[0]+=64;v_f32_st_tnsr(p,output,hi##R,0,row+R<rows);}
#endif
   STORE(0)
#if ROWS == 4
   STORE(1) STORE(2) STORE(3)
#endif
#undef STORE
  }
 }
}
