// FP32 reduction and optional FP32 bias, with exactly one final BF16 conversion.
void main(tensor input,
#if BIAS
 tensor bias,
#endif
 tensor output) {
 int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
#if REDUCE
 int groups=get_dim_size(input,2);
#else
 int groups=1;
#endif
 for(int row=begin[1];row<end[1];++row)for(int nb=begin[0];nb<end[0];++nb){
  float64 lo=0,hi=0;
  for(int g=0;g<groups;++g){int5 p={nb*128,row,g,0,0};lo+=v_f32_ld_tnsr_b(p,input);p[0]+=64;hi+=v_f32_ld_tnsr_b(p,input);}
#if BIAS
  int5 b={nb*128,0,0,0,0};lo+=v_f32_ld_tnsr_b(b,bias);b[0]+=64;hi+=v_f32_ld_tnsr_b(b,bias);
#endif
  int5 p={nb*128,row,0,0,0};
#if OUTPUT_BF16
  float128 pair={lo,hi};v_bf16_st_tnsr(p,output,convert_float128_to_bfloat128(pair,SW_LINEAR|SW_RHNE));
#else
  v_f32_st_tnsr(p,output,lo);p[0]+=64;v_f32_st_tnsr(p,output,hi);
#endif
 }
}
