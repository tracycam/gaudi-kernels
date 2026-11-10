// Per-token activation and per-channel weight scales, FP32 accumulators.
void main(tensor partial, tensor activation_scales, tensor weight_scales, tensor bias, tensor output) {
 int5 begin=get_index_space_offset(), end=begin+get_index_space_size();
 for (int row=begin[1]; row<end[1]; ++row) {
  int5 sp={0,row,0,0,0};
  float64 act=v_f32_ld_g(gen_addr(sp,activation_scales));
  for (int block=begin[0]; block<end[0]; ++block) {
   int col=block*128;
   int5 p0={col,row,0,0,0}, p1={col+64,row,0,0,0};
   int5 w0={col,0,0,0,0}, w1={col+64,0,0,0,0};
   float64 x0=v_f32_ld_tnsr_b(p0,partial)*act, x1=v_f32_ld_tnsr_b(p1,partial)*act;
   float64 s0=v_f32_ld_tnsr_b(w0,weight_scales), s1=v_f32_ld_tnsr_b(w1,weight_scales);
   float64 b0=v_f32_ld_tnsr_b(w0,bias), b1=v_f32_ld_tnsr_b(w1,bias);
   float128 result={v_f32_mac_b(x0,s0,b0),v_f32_mac_b(x1,s1,b1)};
   v_bf16_st_tnsr(p0,output,convert_float128_to_bfloat128(result,SW_LINEAR|SW_RHNE));
  }
 }
}
