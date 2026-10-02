#ifndef A8
#define A8 0
#endif
#if A8
void main(tensor input,tensor activation_scales,tensor weight_scales,tensor bias,tensor output) {
#else
void main(tensor input,tensor weight_scales,tensor bias,tensor output) {
#endif
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 for(int row=begin[1];row<end[1];++row)for(int nb=begin[0];nb<end[0];++nb) {
  int5 p={nb*128,row,0,0,0},sp={nb*128,0,0,0,0};
#if A8
  int5 ap={0,row,0,0,0};float64 as=v_f32_ld_g(gen_addr(ap,activation_scales));
#else
  float64 as=1.f;
#endif
  float64 lo=v_f32_ld_tnsr_b(p,input)*v_f32_ld_tnsr_b(sp,weight_scales)*as+v_f32_ld_tnsr_b(sp,bias);
  p[0]+=64;sp[0]+=64;
  float64 hi=v_f32_ld_tnsr_b(p,input)*v_f32_ld_tnsr_b(sp,weight_scales)*as+v_f32_ld_tnsr_b(sp,bias);
  p[0]-=64;float128 pair={lo,hi};
  v_bf16_st_tnsr(p,output,convert_float128_to_bfloat128(pair,SW_LINEAR|SW_RHNE));
 }
}
