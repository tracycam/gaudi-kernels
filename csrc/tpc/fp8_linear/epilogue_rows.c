// Four rows share channel scales/bias. Each access pattern covers all four.
#if A8
void main(tensor input,tensor activation_scales,tensor weight_scales,tensor bias,tensor output) {
#else
void main(tensor input,tensor weight_scales,tensor bias,tensor output) {
#endif
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 int rows=get_dim_size(output,1);
 for(int nb=begin[0];nb<end[0];++nb) {
  int5 sp={nb*128,0,0,0,0};
  float64 wl=v_f32_ld_tnsr_b(sp,weight_scales),bl=v_f32_ld_tnsr_b(sp,bias);sp[0]+=64;
  float64 wh=v_f32_ld_tnsr_b(sp,weight_scales),bh=v_f32_ld_tnsr_b(sp,bias);
  for(int rb=begin[1];rb<end[1];++rb) {
   int row=rb*4;
#if A8
#define SCALE(R) int5 ap##R={0,row+R,0,0,0};float64 as##R=v_f32_ld_g(gen_addr(ap##R,activation_scales),0,0,row+R<rows);
#else
#define SCALE(R) float64 as##R=1.f;
#endif
#define STEP(R) { \
   SCALE(R) \
   int5 p={nb*128,row+R,0,0,0}; \
   float64 lo=v_f32_ld_tnsr_b(p,input,0,0,row+R<rows)*wl*as##R+bl;p[0]+=64; \
   float64 hi=v_f32_ld_tnsr_b(p,input,0,0,row+R<rows)*wh*as##R+bh;p[0]-=64; \
   float128 pair={lo,hi};v_bf16_st_tnsr(p,output,convert_float128_to_bfloat128(pair,SW_LINEAR|SW_RHNE),0,row+R<rows); \
  }
   STEP(0) STEP(1) STEP(2) STEP(3)
#undef STEP
#undef SCALE
  }
 }
}
